# paperModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F 

class PaperMultiScaleCNN(nn.Module):
   
    def __init__(self, num_channels=14, num_classes=2, sampling_rate=128):
        super().__init__()
        
        # Temporal kernel ratios 
        self.ratios = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        window_length = 128
        self.kernel_sizes = [max(1, int(r * window_length)) for r in self.ratios]
        
        print(f"Temporal kernel sizes: {self.kernel_sizes}")
        
        filters_per_branch = 8
        
        # TEMPORAL LAYERS
        
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filters_per_branch,
                    kernel_size=(1, k),
                    padding=(0, k // 2)
                ),
                nn.BatchNorm2d(filters_per_branch),
                nn.LeakyReLU(0.01),
                nn.AvgPool2d(kernel_size=(1, 2))
            )
            for k in self.kernel_sizes
        ])
        
        temporal_out_channels = filters_per_branch * len(self.kernel_sizes) 
        
        # SPATIAL LAYERS
        spatial_filters = 8
        
        # Global: (14, 1) → output: (B, 8, 1, T)
        self.global_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_out_channels,
                out_channels=spatial_filters,
                kernel_size=(num_channels, 1)
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        
        # Hemisphere: (7, 1) stride (7, 1) → output: (B, 8, 2, T)
        self.hemisphere_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_out_channels,
                out_channels=spatial_filters,
                kernel_size=(num_channels // 2, 1),
                stride=(num_channels // 2, 1)
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        
        # Quarter: (3, 1) stride (3, 1) → output: (B, 8, 4, T)
        quarter_size = num_channels // 4
        self.quarter_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_out_channels,
                out_channels=spatial_filters,
                kernel_size=(quarter_size, 1),
                stride=(quarter_size, 1)
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        
        
        fusion_input_channels = spatial_filters * 3  # 24
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=fusion_input_channels,
                out_channels=32,
                kernel_size=(8, 1),  # (8,1)
                padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        
        # CLASSIFIER
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        
        # Temporal processing
        temporal_features = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(temporal_features, dim=1)  # (B, 40, 14, T/2)
        
        # Spatial processing
        g = self.global_spatial(x)      # (B, 8, 1, T/4)
        h = self.hemisphere_spatial(x)  # (B, 8, 2, T/4)
        q = self.quarter_spatial(x)     # (B, 8, 4, T/4)
        
        # Get dimensions
        B, C, _, T = g.shape
        
        # Flatten spatial dimension for each branch
        g = g.view(B, C, -1, T)  # (B, 8, 1, T/4)
        h = h.view(B, C, -1, T)  # (B, 8, 2, T/4)
        q = q.view(B, C, -1, T)  # (B, 8, 4, T/4)
        
        # Total: 1 + 2 + 4 = 7, so pad by 1 to get 8
        g_pad = F.pad(g, (0, 0, 0, 7))  # Pad to (B, 8, 8, T/4)
        h_pad = F.pad(h, (0, 0, 0, 6))  # Pad to (B, 8, 8, T/4)
        q_pad = F.pad(q, (0, 0, 0, 4))  # Pad to (B, 8, 8, T/4)
        
        # Concatenate along channel dimension
        x = torch.cat([g_pad, h_pad, q_pad], dim=1)  # (B, 24, 8, T/4)
        
        # Fusion with (8,1) kernel as specified in paper
        x = self.fusion_layer(x)  # (B, 32, 1, T/8)
        
        # Classification
        out = self.classifier(x)  # (B, 2)
        
        return out



# TEST
if __name__ == '__main__':
    model = PaperMultiScaleCNN(num_channels=14, num_classes=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    dummy_input = torch.randn(64, 1, 14, 128)
    output = model(dummy_input)
    
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
