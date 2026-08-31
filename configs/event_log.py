"""
Constants for the event log schema written by the `probabilistic-suffix-prediction` preprocessing
pipeline: the `train.csv` / `val.csv` / `test.csv` files `event_log_loader/reader.py` reads.
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

# Where the codec declares those same two columns. The time till next event has a key of its own,
# while the time since case start is one of the event features the preprocessing pipeline derives
# from the timestamp, and so sits inside `numeric_features` with the calendar features. Only the
# codec's layout is meant here: the columns it names are still 'ts_prev' and 'ts_start', which is
# what `COLUMN_RENAMES` keys off.
TIME_TO_NEXT_KEY = 'time_to_next'
CASE_ELAPSED_KEY = 'ts_start'

# The remaining time to the end of the case. It is what the suffix predicts, so it must never be
# read as an input feature.
REMAINING_TIME_COLUMN = 'rtime'

# Value written into every categorical column of the end-of-sequence events, as in
# `CSV2EventLog._CSV2EventLog__add_last_rows`.
EOS_LABEL = 'EOS'

# What a categorical value the codec's vocabulary does not hold is encoded as, and read back as.
# Spelled the way the preprocessing pipeline spells it, so an unknown value compares equal across
# models.
UNKNOWN_LABEL = 'UNK'

SPLITS = ('train', 'val', 'test')
