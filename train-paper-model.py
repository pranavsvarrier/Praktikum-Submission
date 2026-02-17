import torch
import torch.nn as nn
import numpy as np
from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from torch.utils.data import DataLoader, Subset
from torcheeg.trainers import ClassifierTrainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from dataReshaping import ReorderChannels
from paperModel import PaperMultiScaleCNN

emotion = 'dominance'


class ValidationCallback(Callback):
    
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


# DATASET WITH CHANNEL REORDERING


dataset = DREAMERDataset(
    io_path='./dreamer_cache',
    mat_path=None,
    offline_transform=None,
    online_transform=transforms.Compose([
        transforms.ToTensor(),
        ReorderChannels()  # Reorder channels anti-clockwise
    ]),
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

print(f"\nNOTE: Trial-level splitting will have subject leakage!")
print(f"   Total subjects in dataset: {len(np.unique(subject_ids))}")

fold_accuracies = []

for i, (train_idx, test_idx) in enumerate(k_fold.split(all_indices, all_labels)):

    print(f"FOLD {i+1}/5 - TRIAL-LEVEL SPLIT - PAPER MODEL")

    
    # Create datasets
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Check subject leakage
    train_subjects = set(subject_ids[train_idx])
    test_subjects = set(subject_ids[test_idx])
    overlap = train_subjects & test_subjects
    
    print(f"\n  Train: {len(train_dataset):6d} samples ({len(train_subjects)} subjects)")
    print(f"  Test:  {len(test_dataset):6d} samples ({len(test_subjects)} subjects)")
    print(f" Subject overlap: {len(overlap)} subjects")
    
    train_labels_check = [all_labels[idx] for idx in train_idx]
    test_labels_check = [all_labels[idx] for idx in test_idx]
    
    train_counts = Counter(train_labels_check)
    test_counts = Counter(test_labels_check)
    
    print(f"\n Fold {i+1} Class Distributions:")
    print(f"  Train: Class 0={train_counts[0]} ({train_counts[0]/len(train_labels_check)*100:.1f}%), Class 1={train_counts[1]} ({train_counts[1]/len(train_labels_check)*100:.1f}%)")
    print(f"  Test:  Class 0={test_counts[0]} ({test_counts[0]/len(test_labels_check)*100:.1f}%), Class 1={test_counts[1]} ({test_counts[1]/len(test_labels_check)*100:.1f}%)")
    
    #dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = PaperMultiScaleCNN(num_channels=14, num_classes=2)
    
    trainer = ClassifierTrainer(
        model=model,
        num_classes=2,
        lr=1e-4,
        weight_decay=1e-4,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    
    # Train
    print(f"\nTraining Fold {i+1}...")
    trainer.fit(
        train_loader,
        test_loader,
        max_epochs=50,
        default_root_dir=f'./checkpoints_paper_model_trial_5fold/fold_{i+1}',
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            ValidationCallback()
        ],
        enable_progress_bar=True,
        enable_model_summary=True if i == 0 else False,
        limit_val_batches=1.0
    )
    
    # Test
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

print(f"\n{'='*70}")
print(f"FINAL RESULTS - Paper Model {emotion.upper()} - PAPER MODEL (TRIAL-LEVEL 5-FOLD CV)")
print(f"{'='*70}\n")

print("Per-fold accuracies:")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"  Fold {i}: {acc:.4f}")

print(f"\nMean ± Std:")
print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"  Min:      {np.min(fold_accuracies):.4f}")
print(f"  Max:      {np.max(fold_accuracies):.4f}")



os.makedirs('results', exist_ok=True)
results_file = f'results/paper_model_{emotion}_trial_level_5fold.txt'

with open(results_file, 'w') as f:
    f.write(f"Paper Model 5-Fold Cross-Validation Results - {emotion.upper()}\n")
    f.write("="*70 + "\n\n")
    
    f.write("WARNING: TRIAL-LEVEL SPLIT (HAS SUBJECT LEAKAGE)\n")
    f.write("   This replicates the paper's methodology but inflates performance.\n")
    f.write("   Does NOT test generalization to new subjects.\n\n")
    
    f.write("Configuration:\n")
    f.write(f"  Model: PaperMultiScaleCNN\n")
    f.write(f"  Emotion: {emotion}\n")

    
    f.write("Per-fold Results:\n")
    for i, acc in enumerate(fold_accuracies, 1):
        f.write(f"  Fold {i}: {acc:.4f}\n")
    
    f.write(f"\n{'='*70}\n")
    f.write("Summary Statistics:\n")
    f.write("="*70 + "\n")
    f.write(f"  Mean Accuracy: {mean_accuracy:.4f}\n")
    f.write(f"  Std Accuracy:  {std_accuracy:.4f}\n")
    f.write(f"  Min Accuracy:  {np.min(fold_accuracies):.4f}\n")
    f.write(f"  Max Accuracy:  {np.max(fold_accuracies):.4f}\n")

print(f"\nResults saved to: {results_file}")
