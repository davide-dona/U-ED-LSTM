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


def encoded_stem(encoded_dir : str,
                 dataset : str,
                 min_suffix_size : int = MIN_SUFFIX_SIZE) -> str:
    """
    The path every file of one encoded dataset is named from.

    ARGS:
    - encoded_dir: Where the encoded datasets live.
    - dataset: Dataset name.
    - min_suffix_size: The value the dataset was built with, which is part of its name: two datasets
      cut with different values sit side by side.

    OUTPUTS:
    - stem: The path without the `_<split>.pkl` a caller appends.
    """
    return os.path.join(encoded_dir, f'{dataset}_all_{min_suffix_size}')
