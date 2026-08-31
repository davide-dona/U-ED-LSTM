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
    # Epochs without a validation improvement before training stops. None runs all `epochs`, which
    # is what the original repository did; the checkpoint is the best epoch on validation either
    # way, so stopping only saves the time the run would have spent not improving.
    'early_stopping_patience': 10,
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
    # The two bpic datasets train on 700k+ windows each. The step is bound by kernel launches
    # rather than by arithmetic, so a batch four times wider costs almost the same per step and
    # buys four times fewer steps per epoch; the learning rate is scaled linearly with it, from the
    # 1e-6 the original 256-window batch used, to keep the per-sample step size the same.
    'bpic17': COMMON | {
        'optimizer': 'adam',
        'learning_rate': 4e-6,
        'scheduler_factor': 0.1,
        'scheduler_patience': 2,
        'scheduler_min_lr': 1e-10,
        'batch_size': 1024,
    },
    # The same log, filtered down to its declined and rejected cases. Two thirds the windows and
    # half the trace length of bpic17, but the same shape of log, so it trains with bpic17's
    # hyperparameters rather than its own.
    'bpic17-dr': COMMON | {
        'optimizer': 'adam',
        'learning_rate': 4e-6,
        'scheduler_factor': 0.1,
        'scheduler_patience': 2,
        'scheduler_min_lr': 1e-10,
        'batch_size': 1024,
    },
    'bpic19': COMMON | {
        'optimizer': 'adam',
        'learning_rate': 4e-6,
        'scheduler_factor': 0.1,
        'scheduler_patience': 2,
        'scheduler_min_lr': 1e-10,
        'batch_size': 1024,
    },
}

assert set(DATASETS) == set(DATASET_NAMES), \
    f"Every dataset needs hyperparameters: {sorted(set(DATASET_NAMES) - set(DATASETS))} missing"
