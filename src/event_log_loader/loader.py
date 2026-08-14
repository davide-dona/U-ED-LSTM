"""
The read side of the format adapter, end to end.

Plays the role the original `EventLogLoader` played for a raw CSV, minus the feature engineering,
the splitting and the fitting, all three of which the preprocessing pipeline already did:

    codec  -> spec.py      which features exist, and how the windows are shaped
    csv    -> reader.py    the split's own values, with the end-of-sequence events appended
    codec  -> encoding.py  the vocabularies and statistics, and the split as arrays
    arrays -> windows.py   the windows, and the dataset serving them
"""

from pathlib import Path

import numpy as np

from configs.event_log import MIN_PREFIX_COLUMN, SPLITS

from .encoding import FeatureCodec
from .reader import add_eos_events, read_split
from .spec import DatasetSpec
from .windows import WindowDataset, cut_points


class PreSplitEventLogLoader:
    """Builds one `WindowDataset` per split from the preprocessed logs."""

    def __init__(self,
                 spec : DatasetSpec,
                 data_root : str | Path):
        """
        ARGS:
        - spec: The dataset specification, normally from `spec_from_codec`.
        - data_root: The `data/` directory of the preprocessing repository.
        """
        self.spec = spec
        self.codec = FeatureCodec.from_spec(spec, data_root)
        self.all_categories = self.codec.all_categories

        self.splits = {split: add_eos_events(read_split(spec, split, data_root), spec)
                       for split in SPLITS}

    def get_dataset(self, split : str) -> WindowDataset:
        """
        Encode one split and enumerate its windows.

        ARGS:
        - split: One of 'train', 'val', 'test'.

        OUTPUTS:
        - dataset: The split's windows.
        """
        df, case_ids, real_lengths = self.splits[split]
        categorical, continuous = self.codec.encode(df)

        augmented_lengths = real_lengths + self.spec.min_suffix_size
        offsets = np.concatenate(([0], np.cumsum(augmented_lengths)))[:-1]

        # One row per case, taken from the first event of each case.
        min_prefix_lengths = df[MIN_PREFIX_COLUMN].to_numpy()[offsets]

        # The test split is the one both models are scored on, so its windows are restricted to the
        # cut points the other model enumerates. The training and validation splits keep this
        # repository's own range; their bounds are 1 throughout, the out-of-time separation being
        # what raises them.
        cuts = cut_points(self.spec,
                          real_lengths=real_lengths,
                          min_prefix_lengths=min_prefix_lengths,
                          match_reference_cuts=(split == 'test'))

        return WindowDataset(spec=self.spec,
                             categorical=categorical,
                             continuous=continuous,
                             offsets=offsets,
                             case_ids=case_ids,
                             real_lengths=real_lengths,
                             cuts=cuts,
                             all_categories=self.all_categories,
                             codec=self.codec,
                             window_size=self.spec.window_size,
                             suffix_data_split_value=self.spec.suffix_data_split_value)
