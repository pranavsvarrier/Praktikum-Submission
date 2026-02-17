import numpy as np
import os
from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms


# Create output directory
os.makedirs('dreamer_cache_emotions', exist_ok=True)

# #Create cache (only run once)
# print("Creating cache...")
# dataset = DREAMERDataset(
#     io_path='./dreamer_cache',
#     mat_path='data/DREAMER.mat',
#     offline_transform=transforms.Compose([
#         transforms.BaselineRemoval(),
#         transforms.MeanStdNormalize(),
#         transforms.To2d()
#     ]),
#     online_transform=transforms.ToTensor(),
#     label_transform=None,  
#     num_channel=14,
#     chunk_size=128,
#     baseline_chunk_size=128,
#     num_baseline=61,
#     num_worker=4
# )

# print(f"Cache created: {len(dataset)} samples")


#Load Data from Cache for creating seperate datasets for Valence, Arousal and Dominance
dataset_valence = DREAMERDataset(
    io_path='./dreamer_cache',
    mat_path=None,
    offline_transform=None,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(3.0)
    ]),
    num_channel=14,
    chunk_size=128,
    baseline_chunk_size=128,
    num_baseline=61
)

dataset_arousal = DREAMERDataset(
    io_path='./dreamer_cache',
    mat_path=None,
    offline_transform=None,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('arousal'),
        transforms.Binary(3.0)
    ]),
    num_channel=14,
    chunk_size=128,
    baseline_chunk_size=128,
    num_baseline=61
)

dataset_dominance = DREAMERDataset(
    io_path='./dreamer_cache',
    mat_path=None,
    offline_transform=None,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('dominance'),
        transforms.Binary(3.0)
    ]),
    num_channel=14,
    chunk_size=128,
    baseline_chunk_size=128,
    num_baseline=61
)
