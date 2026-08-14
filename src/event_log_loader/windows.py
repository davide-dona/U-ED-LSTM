"""
Cutting an encoded split into the prefix windows the model is fed.

A window of `prefix_len` events is fed to the encoder minus its last `suffix_data_split_value`
positions, so it conditions on the case's first `k = prefix_len - suffix_data_split_value` events.
That `k` is the cut point the other model's `TraceDataset` enumerates.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .encoding import FeatureCodec
from .spec import DatasetSpec


def cut_points(spec : DatasetSpec,
               real_lengths : np.ndarray,
               min_prefix_lengths : np.ndarray,
               match_reference_cuts : bool) -> np.ndarray:
    """
    Enumerate the windows of every case.

    ARGS:
    - spec: The dataset specification.
    - real_lengths: Number of real events per case.
    - min_prefix_lengths: Lower bound on the cut points per case, from the preprocessed log. It is
      above 1 only for the cases crossing the out-of-time separation, whose early prefixes a
      training run could already have seen.
    - match_reference_cuts: If true, emit exactly the cut points `k` in
      `[min_prefix_len, real_length - 1]`, i.e. the other model's test population, one window each.
      If false, use this repository's own range, whose trailing all-EOS targets are what teach the
      model to stop.

    OUTPUTS:
    - cuts: int64, [num_windows, 2], holding (case index, prefix_len).
    """
    split_value = spec.suffix_data_split_value

    if match_reference_cuts:
        first = min_prefix_lengths + split_value
        last = real_lengths - 1 + split_value
    else:
        first = np.maximum(spec.min_suffix_size, min_prefix_lengths + split_value)
        last = real_lengths + spec.min_suffix_size

    counts = np.maximum(last - first + 1, 0)
    case_index = np.repeat(np.arange(len(real_lengths)), counts)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    prefix_len = np.repeat(first, counts) + (np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts))

    assert (prefix_len >= spec.min_suffix_size).all(), \
        "A window shorter than min_suffix_size would leave the decoder a padded target"
    assert (prefix_len <= real_lengths[case_index] + spec.min_suffix_size).all(), \
        "A window cannot reach past the end-of-sequence events"

    return np.stack((case_index, prefix_len), axis=1)


class WindowDataset(Dataset):
    """
    Prefix windows of one split, materialized on access.

    Interchangeable with `EventLogDataset`: it exposes the same `all_categories` and returns the
    same `(categorical, numerical, case_id)` triple, so the trainer and the model take it unchanged.
    Windows are built in `__getitem__` rather than up front because the eager form costs the number
    of windows times the window size: for bpic17 that is about 7 GB of tensors, against 50 MB here.
    """

    def __init__(self,
                 spec : DatasetSpec,
                 categorical : np.ndarray,
                 continuous : np.ndarray,
                 offsets : np.ndarray,
                 case_ids : np.ndarray,
                 real_lengths : np.ndarray,
                 cuts : np.ndarray,
                 all_categories : tuple,
                 codec : FeatureCodec,
                 window_size : int,
                 suffix_data_split_value : int):
        # Kept so that a pickled dataset describes itself: the training script reads `dec_feat` off
        # it rather than being told the column names a second time.
        self.spec = spec
        self.categorical = categorical
        self.continuous = continuous
        self.offsets = offsets
        self.case_ids = case_ids
        self.real_lengths = real_lengths
        self.cuts = cuts
        self.all_categories = all_categories
        self.codec = codec
        self.window_size = window_size
        self.suffix_data_split_value = suffix_data_split_value

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, idx):
        case_index, prefix_len = self.cuts[idx]
        start = self.offsets[case_index]

        cat_window = self._to_window(self.categorical[start : start + prefix_len])
        cont_window = self._to_window(self.continuous[start : start + prefix_len])

        cat = tuple(torch.from_numpy(cat_window[:, i].astype(np.int64))
                    for i in range(cat_window.shape[1]))
        cont = tuple(torch.from_numpy(cont_window[:, i])
                     for i in range(cont_window.shape[1]))

        return (cat, cont, self.case_ids[case_index])

    def _to_window(self, values : np.ndarray) -> np.ndarray:
        """
        Pad on the left with zeros, or keep the last `window_size` events.

        Zero is the padding row for the categorical columns and, the columns being standardized,
        exactly the training mean for the continuous ones.
        """
        if len(values) >= self.window_size:
            return values[len(values) - self.window_size:]

        window = np.zeros((self.window_size, values.shape[1]), dtype=values.dtype)
        window[self.window_size - len(values):] = values
        return window

    def index_frame(self) -> pd.DataFrame:
        """
        Describe every window, in dataset order.

        The evaluation pipeline scores a generated suffix against the prefix it continues, so a
        sampled suffix has to be traceable back to a (case, cut point) pair. That mapping only
        exists here, and is written next to the dataset by `scripts/build_datasets.py`.

        OUTPUTS:
        - frame: One row per window, holding the case, the window length, the cut point the window
          conditions on, and the number of real events left to predict from it.
        """
        case_index, prefix_len = self.cuts[:, 0], self.cuts[:, 1]
        cut_point = prefix_len - self.suffix_data_split_value
        return pd.DataFrame({
            'row': np.arange(len(self.cuts)),
            'case_id': self.case_ids[case_index],
            'prefix_len': prefix_len,
            'cut_point': cut_point,
            'suffix_len': np.maximum(self.real_lengths[case_index] - cut_point, 0),
        })
