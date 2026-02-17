import numpy as np
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from torch.utils.data import Subset
import torch


def trial_level_64_16_20_split(dataset, seed=42):
    
    # Get all indices
    all_indices = np.arange(len(dataset))
    
    # Get labels
    all_labels = []
    for idx in all_indices:
        _, label = dataset[idx]
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    all_labels = np.array(all_labels)
       
    # First split: 80% (train+val) vs 20% (test)
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=all_labels  # ✅ Stratify by class labels
    )
    
    # Get labels for train+val
    train_val_labels = all_labels[train_val_idx]
    
    # Second split: 80% (train) vs 20% (val) of the train+val set
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,  
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels  # Stratify by class labels
    )
    
    #subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    #subject leakage check
    subject_ids = dataset.info['subject_id'].values
    
    train_subjects = set(subject_ids[train_idx])
    val_subjects = set(subject_ids[val_idx])
    test_subjects = set(subject_ids[test_idx])
    
    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects
    
    print(f"\n SUBJECT LEAKAGE CHECK:")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Test subjects:  {len(test_subjects)}")
    print(f"\n Subjects in BOTH train & val:  {len(train_val_overlap)}")
    print(f" Subjects in BOTH train & test: {len(train_test_overlap)}")
    print(f"Subjects in BOTH val & test:   {len(val_test_overlap)}")
    
    if len(train_test_overlap) > 0:
        print(f"\nWARNING: {len(train_test_overlap)} subjects appear in BOTH train and test!")
        print(f"This inflates performance and does NOT test generalization to new subjects.")
    
    return train_dataset, val_dataset, test_dataset



import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Subset
import torch

# Subject wise split without stratification
# def subject_wise_64_16_20_split(dataset, seed=42):
#     """
#     Subject-wise 64-16-20 split (NO SUBJECT LEAKAGE)
#     GroupShuffleSplit to ensure subjects don't overlap between splits.
#     """
    
#     # Get all indices
#     all_indices = np.arange(len(dataset))
    
#     # Get labels for information
#     all_labels = []
#     for idx in all_indices:
#         _, label = dataset[idx]
#         all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
#     all_labels = np.array(all_labels)
    
#     # Get subject IDs (groups)
#     subject_ids = dataset.info['subject_id'].values
    
    
    
#     print(f"  Total subjects: {len(np.unique(subject_ids))}")
    
#     # First split: 80% (train+val) vs 20% (test) - GROUP BY SUBJECT
#     gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
#     train_val_idx, test_idx = next(gss_test.split(all_indices, all_labels, groups=subject_ids))
    
#     #Second split: 80% (train) vs 20% (val) of train+val - GROUP BY SUBJECT
#     # 64% train, 16% val, 20% test overall
#     train_val_labels = all_labels[train_val_idx]
#     train_val_subjects = subject_ids[train_val_idx]
    
#     gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
#     train_idx_local, val_idx_local = next(gss_val.split(
#         train_val_idx, 
#         train_val_labels, 
#         groups=train_val_subjects
#     ))
    
#     # Convert local indices back to global indices
#     train_idx = train_val_idx[train_idx_local]
#     val_idx = train_val_idx[val_idx_local]
    
#     # Create subsets
#     train_dataset = Subset(dataset, train_idx)
#     val_dataset = Subset(dataset, val_idx)
#     test_dataset = Subset(dataset, test_idx)
    
#     #NO subject leakage Verification
#     train_subjects = set(subject_ids[train_idx])
#     val_subjects = set(subject_ids[val_idx])
#     test_subjects = set(subject_ids[test_idx])
    
#     train_val_overlap = train_subjects & val_subjects
#     train_test_overlap = train_subjects & test_subjects
#     val_test_overlap = val_subjects & test_subjects
    
#     print(f"\nSUBJECT LEAKAGE CHECK:")
#     print(f"  Train subjects: {len(train_subjects)}")
#     print(f"  Val subjects:   {len(val_subjects)}")
#     print(f"  Test subjects:  {len(test_subjects)}")
#     print(f"\n  Subjects in BOTH train & val:  {len(train_val_overlap)}")
#     print(f"  Subjects in BOTH train & test: {len(train_test_overlap)}")
#     print(f"  Subjects in BOTH val & test:   {len(val_test_overlap)}")
    
#     if len(train_test_overlap) == 0 and len(train_val_overlap) == 0 and len(val_test_overlap) == 0:
#         print(f"\nNo subject leakage detected")
        
#     else:
#         print(f"\nSubject leakage detected")
    
#     # Print sample counts
#     print(f"\nSample Distribution:")
#     print(f"  Train: {len(train_dataset):6d} samples ({len(train_subjects)} subjects)")
#     print(f"  Val:   {len(val_dataset):6d} samples ({len(val_subjects)} subjects)")
#     print(f"  Test:  {len(test_dataset):6d} samples ({len(test_subjects)} subjects)")
    
#     return train_dataset, val_dataset, test_dataset

# Subject wise split with stratification
def subject_wise_64_16_20_split(dataset, seed=42):

    
    all_indices = np.arange(len(dataset))
    
    #labels
    all_labels = []
    for idx in all_indices:
        _, label = dataset[idx]
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    all_labels = np.array(all_labels)
    
    #subject IDs
    subject_ids = dataset.info['subject_id'].values
    unique_subjects = np.unique(subject_ids)
    
    print(f"  Total subjects: {len(unique_subjects)}")
    
    #Stratification
    subject_strata = {}
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        subject_labels = all_labels[subject_mask]
        ratio_class1 = np.mean(subject_labels)
        
        # Bin into quartiles
        if ratio_class1 < 0.25:
            strata = 0 
        elif ratio_class1 < 0.50:
            strata = 1 
        elif ratio_class1 < 0.75:
            strata = 2  
        else:
            strata = 3
        
        subject_strata[subject] = strata
    
    
    strata_names = {0: 'very_low', 1: 'low', 2: 'high', 3: 'very_high'}
    print(f"\n  Subject Class Ratio Strata:")
    for s_id, name in strata_names.items():
        count = sum(1 for v in subject_strata.values() if v == s_id)
        print(f"    {name:10s}: {count} subjects")
    
    
    trial_strata = np.array([subject_strata[s] for s in subject_ids])
    
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    
    #80% train+val, 20% test
    train_val_idx, test_idx = next(sgkf.split(
        all_indices,
        trial_strata,       # Stratify by subject class ratio
        groups=subject_ids  
    ))
    
    
    train_val_strata = trial_strata[train_val_idx]
    train_val_subjects = subject_ids[train_val_idx]
    
    # n_splits=5 gives 80/20 split of remaining train+val
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    
    train_idx_local, val_idx_local = next(sgkf_val.split(
        train_val_idx,
        train_val_strata,       # Stratify by subject class ratio
        groups=train_val_subjects
    ))
    
    #local indices back to global indices
    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]
    
    #subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
   
    train_subjects = set(subject_ids[train_idx])
    val_subjects = set(subject_ids[val_idx])
    test_subjects = set(subject_ids[test_idx])
    
    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects
    
    print(f"\nSUBJECT LEAKAGE CHECK:")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects:   {len(val_subjects)}")
    print(f"  Test subjects:  {len(test_subjects)}")
    print(f"\n  Subjects in BOTH train & val:  {len(train_val_overlap)}")
    print(f"  Subjects in BOTH train & test: {len(train_test_overlap)}")
    print(f"  Subjects in BOTH val & test:   {len(val_test_overlap)}")
    
    if len(train_test_overlap) == 0 and len(train_val_overlap) == 0 and len(val_test_overlap) == 0:
        print(f"\nNo subject leakage detected")
    else:
        print(f"\nSubject leakage detected")
    
    # Class distribution check
    print(f"\nClass Distribution Check:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        labels = all_labels[idx]
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        total = len(labels)
        print(f"  {name}: Class 0={class_dist.get(0,0)} ({class_dist.get(0,0)/total*100:.1f}%), "
              f"Class 1={class_dist.get(1,0)} ({class_dist.get(1,0)/total*100:.1f}%)")
    
    # Sample counts
    print(f"\nSample Distribution:")
    print(f"  Train: {len(train_dataset):6d} samples ({len(train_subjects)} subjects)")
    print(f"  Val:   {len(val_dataset):6d} samples ({len(val_subjects)} subjects)")
    print(f"  Test:  {len(test_dataset):6d} samples ({len(test_subjects)} subjects)")
    
    return train_dataset, val_dataset, test_dataset