class ReorderChannels:
  
    def __init__(self):
        # DREAMER channel order (indices 0-13)
        dreamer_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                           'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        # Anti-clockwise order
        anticlockwise_order = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1',
                              'O2', 'P8', 'T8', 'FC6', 'F8', 'F4', 'AF4']
        
        # Create mapping
        self.reorder_indices = [dreamer_channels.index(ch) for ch in anticlockwise_order]
        
        print(f"Channel reordering map: {dreamer_channels} → {anticlockwise_order}")
        print(f"Indices: {self.reorder_indices}")
    
    def __call__(self, **kwargs):
    
        eeg = kwargs['eeg']
  
        if eeg.dim() == 3:
            # Shape: (1, 14, 128) -> reorder along dimension 1
            eeg_reordered = eeg[:, self.reorder_indices, :]
        elif eeg.dim() == 2:
            # Shape: (14, 128) -> reorder along dimension 0
            eeg_reordered = eeg[self.reorder_indices, :]
        else:
            raise ValueError(f"Bad EEG shape: {eeg.shape}")
        
        # Return dict with all original keys, only 'eeg' modified
        result = kwargs.copy()
        result['eeg'] = eeg_reordered
        
        return result