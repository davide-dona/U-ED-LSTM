"""
Which datasets the adapter supports, and the window geometry every one of them is cut with.

The geometry lives here because everything downstream has to agree on it: `build_datasets.py` cuts
the windows with it, `configs/training.py` shapes the decoder from it, and `train.py` and
`generate.py` find the encoded files by the name it gives them. Changing either number invalidates
the encoded datasets and every checkpoint trained on them, so it is set once, here, rather than
defaulted separately in each script.
"""

import os

# The datasets, i.e. the directories under the preprocessing repository's `data/`.
DATASETS = ('sepsis', 'bpic17', 'bpic19')

# End-of-sequence events appended to every case. A window has to be at least this long, so that the
# decoder's target is never padding, and a case's last window reaches exactly this far past its last
# real event, which is what teaches the model to stop.
MIN_SUFFIX_SIZE = 5

# Trailing window positions the decoder is trained to predict, and hence how far behind the end of a
# window the prefix it conditions on ends. This is the model's `seq_len_pred` and the loader's
# `suffix_data_split_value`: one number under two names, because the model and the windows have to
# be built from the same one.
SEQ_LEN_PRED = 4

# Where `build_datasets.py` writes the encoded datasets and the other two scripts read them.
ENCODED_DIR = 'encoded_data'

# Where `train.py` writes a trained model and `generate.py` reads it back.
CHECKPOINT_DIR = 'checkpoints'


def encoded_stem(dataset : str) -> str:
    """
    The path every file of one encoded dataset is named from.

    `MIN_SUFFIX_SIZE` is part of the name, so datasets cut with different values would sit side by
    side; the value is read from this module rather than passed, so the three scripts cannot disagree
    on which of them they are looking at.

    ARGS:
    - dataset: Dataset name.

    OUTPUTS:
    - stem: The path without the `_<split>.pkl` a caller appends.
    """
    return os.path.join(ENCODED_DIR, f'{dataset}_all_{MIN_SUFFIX_SIZE}')


def checkpoint_path(dataset : str) -> str:
    """
    Where one dataset's trained model lives.

    ARGS:
    - dataset: Dataset name.

    OUTPUTS:
    - path: The file `train.py` saves to and `generate.py` loads from.
    """
    return os.path.join(CHECKPOINT_DIR, f'{dataset}_u_ed_lstm.pkl')
