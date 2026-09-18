"""
Constants for the event log schema written by the `suffix-generation` preprocessing
pipeline: the `train.csv` / `val.csv` / `test.csv` files `event_log_loader/reader.py` reads.
"""

# Canonical column names written by the external preprocessing pipeline.
CASE_COLUMN = 'case:concept:name'
MIN_PREFIX_COLUMN = 'min_prefix_len'
CSV_SEPARATOR = ';'

# The upstream preprocessing contract exposes the two durations under these canonical names. The
# values stay in minutes; every numerical column is standardized, so the unit is irrelevant.
CASE_ELAPSED_COLUMN = 'ts_start'
EVENT_ELAPSED_COLUMN = 'inter_event_time'

# The time since case start is an event feature; the wait before an event has its own codec entry.
CASE_ELAPSED_KEY = 'ts_start'
INTER_EVENT_TIME_KEY = 'inter_event_time'

# The remaining time to the end of the case. It is what the suffix predicts, so it must never be
# read as an input feature.
REMAINING_TIME_COLUMN = 'rtime'

# Value written into every categorical column of the end-of-trace events.
EOT_LABEL = 'EOT'

# What a categorical value the codec's vocabulary does not hold is encoded as, and read back as.
# Spelled the way the preprocessing pipeline spells it, so an unknown value compares equal across
# models.
UNKNOWN_LABEL = 'UNK'

SPLITS = ('train', 'val', 'test')
