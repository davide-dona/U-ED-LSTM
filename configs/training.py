"""
Per-dataset model and training hyperparameters, retrieved from the original repository.
BPIC19 was not included in the original repository, added for completeness and matching the other BPIC datasets.
"""

from configs.data import DATASETS as DATASET_NAMES
from configs.data import SEQ_LEN_PRED

# Shared by every dataset upstream, on all of Helpdesk, Sepsis and BPIC17.
COMMON = {
    # Decoder output sequence length. The trainer takes the same number of trailing window
    # positions as the target, so this is the geometry the datasets were cut with, not a free
    # hyperparameter: it is read from `configs/data.py` rather than repeated here.
    'seq_len_pred': SEQ_LEN_PRED,
    'hidden_size': 128,
    'num_layers': 4,
    'dropout': 0.1,
    'epochs': 100,
    'regularization_term': 1e-4,
    'teacher_forcing_ratio': 0.8,
    'shuffle': True,
    # GradNorm balances one loss per decoder feature. The number of tasks follows `dec_feat`, so
    # the training script fills it in rather than it being declared here.
    'gn_alpha': 1.5,
    'gn_learning_rate': 1e-4,
}

DATASETS = {
    'sepsis': COMMON | {
        'optimizer': 'adamw',
        'learning_rate': 1e-5,
        'scheduler_factor': 0.01,
        'scheduler_patience': 1,
        'scheduler_min_lr': 1e-9,
        'batch_size': 128,
    },
    'bpic17': COMMON | {
        'optimizer': 'adam',
        'learning_rate': 1e-6,
        'scheduler_factor': 0.1,
        'scheduler_patience': 2,
        'scheduler_min_lr': 1e-10,
        'batch_size': 256,
    },
    'bpic19': COMMON | {
        'optimizer': 'adam',
        'learning_rate': 1e-6,
        'scheduler_factor': 0.1,
        'scheduler_patience': 2,
        'scheduler_min_lr': 1e-10,
        'batch_size': 256,
    },
}

assert set(DATASETS) == set(DATASET_NAMES), \
    f"Every dataset needs hyperparameters: {sorted(set(DATASET_NAMES) - set(DATASETS))} missing"
