EEG-Based Emotion Recognition - Praktikum 2
Pranav Suresh Varrier | 12350534
Analysis and replication of "Consumer-friendly EEG-based Emotion Recognition System: A Multi-scale Convolutional Neural Network Approach"

Requirements
pip install torcheeg pytorch-lightning scikit-learn scipy torch numpy

Dataset Setup
The DREAMER dataset .mat file must be placed in the project directory before running anything. On first run, TorchEEG will preprocess and cache it automatically into ./dreamer_cache. This only happens once.
Data Processing 
Run DataProcessing with code marked with "#Create cache (only run once)" uncommented. This will create a datacache. After that, if you want to use Dataprocessing again, you can comment that code out. But Dataprocessing only needs to be run once. 
Project Structure
 paperModel.py                          # Multi-Scale CNN (paper's model)
 dataReshaping.py                       # ReorderChannels transform (anti-clockwise)
 subjectWiseTrainValSplit.py            # Splitting utility functions
 train_paper_model_train_val_test.py    # Paper model - train/val/test split
 train-paper-model.py                   # Paper model - 5-fold CV
 train_tsception_train_val_test.py      # TSCeption - train/val/test split
 train-tsception-model.py               # TSCeption - 5-fold CV
 train_eegnet_train_val_test.py         # EEGNet - train/val/test split
 train_eegnet_5fold.py                  # EEGNet - 5-fold CV
 dreamer_cache/                         # Cached preprocessed dataset (auto-generated)
 results/                               # Output results (.txt files)

