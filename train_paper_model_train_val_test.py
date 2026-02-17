import torch
import torch.nn as nn
import numpy as np
from torcheeg.datasets import DREAMERDataset
from sklearn.metrics import confusion_matrix
from paperModel import PaperMultiScaleCNN
from torcheeg import transforms
from torch.utils.data import DataLoader, Subset
from torcheeg.trainers import ClassifierTrainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
from subjectWiseTrainValSplit import subject_wise_64_16_20_split,trial_level_64_16_20_split
from dataReshaping import ReorderChannels


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





def calculate_all_metrics(model, data_loader, device):
    """Calculate all 7 metrics with detailed confusion matrix"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            eeg, labels = batch
            eeg = eeg.to(device)
            
            outputs = model(eeg)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"\nDetailed Confusion Matrix:")
    print(f"  True Positives  (TP): {tp:6d}")
    print(f"  True Negatives  (TN): {tn:6d}")
    print(f"  False Positives (FP): {fp:6d}")
    print(f"  False Negatives (FN): {fn:6d}")
    
    total = len(y_true)
    print(f"\nPrediction Distribution:")
    print(f"  Predicted Positive: {tp + fp:6d} ({(tp+fp)/total*100:.1f}%)")
    print(f"  Predicted Negative: {tn + fn:6d} ({(tn+fn)/total*100:.1f}%)")
    
    print(f"\nActual Label Distribution:")
    print(f"  Actual Positive: {tp + fn:6d} ({(tp+fn)/total*100:.1f}%)")
    print(f"  Actual Negative: {tn + fp:6d} ({(tn+fp)/total*100:.1f}%)")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0.0
    
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auroc = 0.5
    
    kappa_denominator = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    kappa = 2 * (tp * tn - fp * fn) / kappa_denominator if kappa_denominator > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'mcc': mcc,
        'auroc': auroc,
        'kappa': kappa,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }


def train_paper_model(emotion='valence'):

    print(f"TRAINING PAPER MODEL - TRIAL-LEVEL 64-16-20 SPLIT - {emotion.upper()}")

    # Load dataset WITH CHANNEL REORDERING
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
    
    # Trial-level split
    
    print("Creating trial-level 64-16-20 split...")
    train_dataset, val_dataset, test_dataset = subject_wise_64_16_20_split(
        dataset=dataset,
        seed=42
    )
    
    #Subject level Split
    # print("Creating trial-level 64-16-20 split...")
    # train_dataset, val_dataset, test_dataset = subject_wise_64_16_20_split(
    #     dataset=dataset,
    #     seed=42
    # )
    
    print(f"\nSample counts:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    #dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
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
    print("TRAINING PAPER MODEL")
    
    trainer.fit(
        train_loader,
        val_loader,
        max_epochs=50,
        default_root_dir=f'./checkpoints_paper_model_{emotion}',
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            ValidationCallback()
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_val_batches=1.0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Evaluate
    print("VALIDATION SET RESULTS")
    val_metrics = calculate_all_metrics(model, val_loader, device)
    
    print(f"\nMetrics:")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  MCC:       {val_metrics['mcc']:.4f}")
    print(f"  AUROC:     {val_metrics['auroc']:.4f}")
    print(f"  Kappa:     {val_metrics['kappa']:.4f}")
    

    print("TEST SET RESULTS")
    test_metrics = calculate_all_metrics(model, test_loader, device)
    
    print(f"\nMetrics:")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  Kappa:     {test_metrics['kappa']:.4f}")
    
    os.makedirs('results', exist_ok=True)
    results_file = f'results/paper_model_{emotion}_train_test_val_trial_level.txt'
    
    with open(results_file, 'w') as f:
        f.write(f"Paper Model Results (Trial-Level 64-16-20 Split) - {emotion.upper()}\n")
        f.write(f"Trial-level split (has subject leakage)\n\n")
        f.write(f"Validation: Acc={val_metrics['accuracy']:.4f}, MCC={val_metrics['mcc']:.4f}, AUROC={val_metrics['auroc']:.4f}\n")
        f.write(f"Test:       Acc={test_metrics['accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, AUROC={test_metrics['auroc']:.4f}\n")
    
    print(f"\n Results saved to: {results_file}")


if __name__ == '__main__':
    EMOTION = 'dominance'
    train_paper_model(emotion=EMOTION)