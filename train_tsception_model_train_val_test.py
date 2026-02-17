from torch.utils.data import DataLoader
from torcheeg.models import TSCeption
from torcheeg.trainers import ClassifierTrainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torcheeg.datasets import DREAMERDataset
from torcheeg import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import torch
import os
from collections import Counter
from subjectWiseTrainValSplit import trial_level_64_16_20_split, subject_wise_64_16_20_split
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

def calculate_all_metrics(model, data_loader, device):
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
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\nDetailed Confusion Matrix:")
    print(f"  True Positives  (TP): {tp:6d}")
    print(f"  True Negatives  (TN): {tn:6d}")
    print(f"  False Positives (FP): {fp:6d}")
    print(f"  False Negatives (FN): {fn:6d}")
    
    print(f"\nPrediction Distribution:")
    total = len(y_true)
    print(f"  Predicted Positive: {tp + fp:6d} ({(tp+fp)/total*100:.1f}%)")
    print(f"  Predicted Negative: {tn + fn:6d} ({(tn+fn)/total*100:.1f}%)")
    
    print(f"\nActual Label Distribution:")
    print(f"  Actual Positive: {tp + fn:6d} ({(tp+fn)/total*100:.1f}%)")
    print(f"  Actual Negative: {tn + fp:6d} ({(tn+fp)/total*100:.1f}%)")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0.0
    
    try:
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auroc = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) > 0 and (tn + fp) > 0 else 0.5
    
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


def train_tsception_64_16_20_stratified(emotion='valence'):
    print(f"TRAINING TSCEPTION - STRATIFIED 64-16-20 SPLIT - {emotion.upper()}")
    
    # Load dataset
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
    
    #subject IDs
    subject_ids = dataset.info['subject_id'].values
    
    print("Creating STRATIFIED Trial Based 64-16-20 split...")
    train_dataset, val_dataset, test_dataset = subject_wise_64_16_20_split(
        dataset=dataset,
        seed=42
    )
    
     
    #STRATIFIED SPLIT TRIAL WISE SPLIT
    # print("Creating STRATIFIED Trial Based 64-16-20 split...")
    # train_dataset, val_dataset, test_dataset = trial_level_64_16_20_split(
    #     dataset=dataset,
    #     seed=42
    # )

    
    print(f"\nSample counts:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Initialize model
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
    
    # Initialize trainer
    trainer = ClassifierTrainer(
        model=model,
        num_classes=2,
        lr=1e-4,
        weight_decay=1e-4,
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    
    # Train
    trainer.fit(
        train_loader,
        val_loader,
        max_epochs=50,
        default_root_dir=f'./checkpoints_{emotion}_64_16_20_stratified',
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            ValidationCallback()
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_val_batches=1.0
    )
    
    # Get device
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
    results_file = f'results/tsception_{emotion}_64_16_20_stratified.txt'
    
    with open(results_file, 'w') as f:
        f.write(f"TSCeption Results (STRATIFIED 64-16-20 Split) - {emotion.upper()}\n")
        f.write(f"Used stratified trial-wise split\n\n")
        f.write(f"Validation: Acc={val_metrics['accuracy']:.4f}, MCC={val_metrics['mcc']:.4f}, AUROC={val_metrics['auroc']:.4f}\n")
        f.write(f"Test:       Acc={test_metrics['accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, AUROC={test_metrics['auroc']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    EMOTION = 'valence'  # Change to 'arousal' or 'dominance'
    train_tsception_64_16_20_stratified(emotion=EMOTION)