from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torcheeg.models import TSCeption
from torcheeg.trainers import ClassifierTrainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter

emotion = 'dominance'

print(f"Training for {emotion}")

class ValidationCallback(Callback):
    """Print validation accuracy after each epoch."""
    
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        
        epoch = trainer.current_epoch
        train_acc = metrics.get('train_accuracy', 0.0)
        train_loss = metrics.get('train_loss', 0.0)
        val_acc = metrics.get('val_accuracy', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        
        print(f"\nEpoch {epoch:2d} Results:")
        print(f"  Train: accuracy={train_acc:.4f}, loss={train_loss:.4f}")
        print(f"  Val:   accuracy={val_acc:.4f}, loss={val_loss:.4f}")

#Load Dataset
dataset = DREAMERDataset(
    io_path='./dreamer_cache',
    mat_path=None,
    offline_transform=None,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select(emotion),
        transforms.Binary(3.0)
    ]),
    num_channel=14,
    chunk_size=128,
    baseline_chunk_size=128,
    num_baseline=61
)


all_indices = np.arange(len(dataset))
all_labels = []

print("\nCollecting labels...")
for idx in all_indices:
    _, label = dataset[idx]
    all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
all_labels = np.array(all_labels)


#5-FOLD TRIAL LEVEL SPLIT
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

subject_ids = dataset.info['subject_id'].values

print(f"   Total subjects in dataset: {len(np.unique(subject_ids))}")

fold_accuracies = []

for i, (train_idx, test_idx) in enumerate(k_fold.split(all_indices, all_labels)):

    print(f"FOLD {i+1}/5 - TRIAL-LEVEL SPLIT")
    
    # Create datasets
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_subjects = set(subject_ids[train_idx])
    test_subjects = set(subject_ids[test_idx])
    overlap = train_subjects & test_subjects
    
    print(f"\n  Train: {len(train_dataset):6d} samples ({len(train_subjects)} subjects)")
    print(f"  Test:  {len(test_dataset):6d} samples ({len(test_subjects)} subjects)")
    print(f"Subject overlap: {len(overlap)} subjects")
    
   
    #dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = TSCeption(
        num_electrodes=14,
        num_classes=2,
        num_T=15,
        num_S=15,
        in_channels=1,
        hid_channels=32,
        sampling_rate=128,
        dropout=0.5
    )

    trainer = ClassifierTrainer(
        model=model,
        num_classes=2,
        lr=1e-4,
        weight_decay=1e-4,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nTraining Fold {i+1}...")
    trainer.fit(
        train_loader,
        test_loader,
        max_epochs=50,
        default_root_dir=f'./checkpoints_trial_5fold/fold_{i+1}',
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            ValidationCallback()
        ],
        enable_progress_bar=True,
        enable_model_summary=True if i == 0 else False,
        limit_val_batches=1.0
    )
    
    score = trainer.test(
        test_loader,
        enable_progress_bar=True,
        enable_model_summary=False
    )[0]
    
    accuracy = score['test_accuracy']
    fold_accuracies.append(accuracy)
    
    print(f"FOLD {i+1} COMPLETE")
    print(f"Final test accuracy: {accuracy:.4f}")
  

mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"FINAL RESULTS - {emotion.upper()} (TRIAL-LEVEL 5-FOLD CV)")

print("Per-fold accuracies:")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"  Fold {i}: {acc:.4f}")

print(f"\nMean ± Std:")
print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"  Min:      {np.min(fold_accuracies):.4f}")
print(f"  Max:      {np.max(fold_accuracies):.4f}")


os.makedirs('results', exist_ok=True)
results_file = f'results/tsception_{emotion}_trial_level_5fold.txt'

with open(results_file, 'w') as f:
    f.write(f"TSCeption 5-Fold Cross-Validation Results - {emotion.upper()}\n")
    
    f.write("Configuration:\n")
    f.write(f"  Model: TSCeption\n")
    f.write(f"  Emotion: {emotion}\n")
    f.write(f"  Learning rate: 1e-4\n")
    f.write(f"  Weight decay: 1e-4\n")
    f.write(f"  Batch size: 64\n")
    f.write(f"  Max epochs: 50\n")
    
    f.write("Per-fold Results:\n")
    for i, acc in enumerate(fold_accuracies, 1):
        f.write(f"  Fold {i}: {acc:.4f}\n")
    
    f.write("Summary Statistics:\n")
    f.write(f"  Mean Accuracy: {mean_accuracy:.4f}\n")
    f.write(f"  Std Accuracy:  {std_accuracy:.4f}\n")
    f.write(f"  Min Accuracy:  {np.min(fold_accuracies):.4f}\n")
    f.write(f"  Max Accuracy:  {np.max(fold_accuracies):.4f}\n")

print(f"\nResults saved to: {results_file}")
