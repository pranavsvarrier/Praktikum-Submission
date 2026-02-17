# EEG-Based Emotion Recognition - Praktikum 2
**Pranav Suresh Varrier | 12350534**

Analysis and replication of "Consumer-friendly EEG-based Emotion Recognition System: A Multi-scale Convolutional Neural Network Approach"

---

## Requirements

```bash
pip install torcheeg pytorch-lightning scikit-learn scipy torch numpy
```

---

## Dataset Setup

The DREAMER dataset `.mat` file must be placed in the project directory before running anything. On first run, set `mat_path='./DREAMER.mat'` in the dataset call. TorchEEG will preprocess and cache it automatically into `./dreamer_cache`. Every run after, set `mat_path=None` as the cache will be detected automatically.

---

## Project Structure

```
├── paperModel.py                          # Multi-Scale CNN (paper's model)
├── dataReshaping.py                       # ReorderChannels transform (anti-clockwise)
├── subjectWiseTrainValSplit.py            # Splitting utility functions
│
├── train_paper_model_train_val_test.py    # Paper model - train/val/test split
├── train-paper-model.py                   # Paper model - 5-fold CV
│
├── train_tsception_train_val_test.py      # TSCeption - train/val/test split
├── train-tsception-model.py               # TSCeption - 5-fold CV
│
├── train_eegnet_train_val_test.py         # EEGNet - train/val/test split
├── train_eegnet_5fold.py                  # EEGNet - 5-fold CV
│
├── dreamer_cache/                         # Cached preprocessed dataset (auto-generated)
├── results/                               # Output results (.txt files)
└── checkpoints_*/                         # Saved model checkpoints
```

---

## Splitting Strategies

All training scripts support two splitting strategies, controlled by commenting/uncommenting inside the script:

| Strategy | Description | Subject Leakage |
|----------|-------------|-----------------|
| Trial-Level + Stratification | Splits individual EEG windows randomly | YES - replicates paper |
| Subject-Wise | Keeps all trials from a subject together | NO - rigorous evaluation |
| Subject-Wise + Stratification  | NO - most rigorous |

---

## Running the Models

All scripts have an `EMOTION` variable at the bottom. Change it to `'valence'`, `'arousal'`, or `'dominance'` before running.

### Paper's Multi-Scale CNN

```bash
# Train/Validation/Test split
python train_paper_model_train_val_test.py

# 5-Fold Cross-Validation
python train-paper-model.py
```

### TSCeption

```bash
# Train/Validation/Test split
python train_tsception_train_val_test.py

# 5-Fold Cross-Validation
python train-tsception-model.py
```

### EEGNet

```bash
# Train/Validation/Test split
python train_eegnet_train_val_test.py

# 5-Fold Cross-Validation
python train_eegnet_5fold.py
```

---

## Switching Between Splitting Strategies

Inside each training script, find the split section and comment/uncomment accordingly:

```python
# OPTION 1: Trial-level split (replicates paper - has subject leakage)
train_dataset, val_dataset, test_dataset = trial_level_64_16_20_split(
    dataset=dataset, seed=42
)

# OPTION 2: Subject-wise split (no leakage)
# train_dataset, val_dataset, test_dataset = subject_wise_64_16_20_split(
#     dataset=dataset, seed=42
# )

# OPTION 3: 
#Comment Stratfied Subject level and uncomment unstraified split to switch between Stratified and unstratified subject splits
```

---

## Results

Results are saved automatically to the `results/` folder after each run as `.txt` files. Naming convention:

```
results/{model}_{emotion}_{split_type}.txt
```

---


## Notes

- The Paper Model requires `ReorderChannels()` transform to arrange EEG channels in anti-clockwise order. This is applied automatically inside the paper model training scripts.
- GPU is used automatically if available, falls back to CPU otherwise.
- Set `num_workers=0` if you encounter multiprocessing errors on Windows.
- All results use `seed=42`.
