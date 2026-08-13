"""
Constants for the event log schema written by the `probabilistic-suffix-prediction` preprocessing
pipeline: the `train.csv` / `val.csv` / `test.csv` files `presplit_loader.py` reads.
"""

# Canonical column names written by the external preprocessing pipeline.
CASE_COLUMN = 'case:concept:name'
MIN_PREFIX_COLUMN = 'min_prefix_len'
CSV_SEPARATOR = ';'

# Time columns, renamed on read to the names this repository's models and checkpoints use.
# The values stay in minutes; every numerical column is standardized, so the unit is irrelevant.
CASE_ELAPSED_COLUMN = 'case_elapsed_time'
EVENT_ELAPSED_COLUMN = 'event_elapsed_time'
COLUMN_RENAMES = {'ts_start': CASE_ELAPSED_COLUMN, 'ts_prev': EVENT_ELAPSED_COLUMN}

# The remaining time to the end of the case. It is what the suffix predicts, so it must never be
# read as an input feature.
REMAINING_TIME_COLUMN = 'rtime'

# Value written into every categorical column of the end-of-sequence events, as in
# `CSV2EventLog._CSV2EventLog__add_last_rows`.
EOS_LABEL = 'EOS'

SPLITS = ('train', 'val', 'test')
