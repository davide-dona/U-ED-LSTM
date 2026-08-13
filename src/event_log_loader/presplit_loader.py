import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .new_event_log_loader import TensorEncoderDecoder
from configs.event_log import (
    CASE_COLUMN,
    CASE_ELAPSED_COLUMN,
    COLUMN_RENAMES,
    CSV_SEPARATOR,
    EOS_LABEL,
    EVENT_ELAPSED_COLUMN,
    MIN_PREFIX_COLUMN,
    REMAINING_TIME_COLUMN,
    SPLITS,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the loader needs to turn one preprocessed dataset into tensors.

    Built by `spec_from_codec` rather than written by hand, so the feature set stays tied to the
    codec the other model was fit against.
    """

    name: str
    case_name: str
    concept_name: str
    resource_name: str
    categorical_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    window_size: int
    min_suffix_size: int
    # Number of trailing window positions the trainer takes as the decoder target, i.e. the model's
    # `seq_len_pred`. A window of `prefix_len` events is the cut point `prefix_len - this`.
    suffix_data_split_value: int

    @property
    def dec_feat(self) -> list[list[str]]:
        """Decoder input and output features: the activity, the resource and the two durations."""
        return [
            [self.concept_name, self.resource_name],
            [CASE_ELAPSED_COLUMN, EVENT_ELAPSED_COLUMN],
        ]

    def as_encoder_kwargs(self) -> dict:
        """Keyword arguments for `TensorEncoderDecoder`."""
        return {
            'case_name': self.case_name,
            'concept_name': self.concept_name,
            'window_size': self.window_size,
            'min_suffix_size': self.min_suffix_size,
            'categorical_columns': list(self.categorical_columns),
            'continuous_columns': list(self.continuous_columns),
        }


def spec_from_codec(dataset: str,
                    data_root : str | Path,
                    min_suffix_size : int = 5,
                    suffix_data_split_value : int = 4) -> DatasetSpec:
    """
    Derive a `DatasetSpec` from the dataset codec of the preprocessing pipeline.

    The codec is the single source of truth for the feature set: it names the activity and resource
    columns and lists exactly the categorical and numerical attributes the other model reads, so
    driving the spec off it keeps the two models on the same features by construction.

    ARGS:
    - dataset: Dataset name, i.e. the directory under `data_root`.
    - data_root: The `data/` directory of the preprocessing repository.
    - min_suffix_size: Number of end-of-sequence events appended to every case.
    - suffix_data_split_value: Number of trailing window positions used as the decoder target.

    OUTPUTS:
    - spec: The dataset specification.
    """
    codec_path = Path(data_root) / dataset / 'codec' / 'dataset.json'
    with open(codec_path) as codec_file:
        codec = json.load(codec_file)

    activity_column = codec['activity']['column']
    resource_column = codec['resource']['column']

    categorical_columns = (activity_column, resource_column) \
        + tuple(feature['column'] for feature in codec['categorical_features'])

    # `numeric_features` already holds the calendar features (day in week, seconds in day), which
    # both models read as standardized numbers rather than as categories.
    continuous_columns = (CASE_ELAPSED_COLUMN, EVENT_ELAPSED_COLUMN) \
        + tuple(feature['column'] for feature in codec['numeric_features'])

    duplicates = [col for col in set(categorical_columns) | set(continuous_columns)
                  if (categorical_columns + continuous_columns).count(col) > 1]
    assert not duplicates, f"Column(s) declared more than once by the codec: {duplicates}"
    assert REMAINING_TIME_COLUMN not in continuous_columns, \
        f"'{REMAINING_TIME_COLUMN}' is the prediction target and must not be an input feature"

    # Every numerical column takes the plain standard-scaling path here; `TensorEncoderDecoder` has
    # no log-scaling path. Fail rather than quietly scale a log-scaled column linearly.
    scaled = [('ts_start', codec['ts_start']), ('ts_prev', codec['ts_prev'])] \
        + [(feature['column'], feature) for feature in codec['numeric_features']]
    log_scaled = [column for column, entry in scaled if entry.get('log')]
    assert not log_scaled, \
        f"The codec log-scales {log_scaled}, which this spec does not support."

    return DatasetSpec(name=dataset,
                       case_name=CASE_COLUMN,
                       concept_name=activity_column,
                       resource_name=resource_column,
                       categorical_columns=categorical_columns,
                       continuous_columns=continuous_columns,
                       # No prefix is ever truncated: the preprocessing pipeline already dropped the
                       # cases longer than `max_trace_length`, and the appended EOS events extend a
                       # full-length case by exactly `min_suffix_size`.
                       window_size=codec['max_trace_length'] + min_suffix_size,
                       min_suffix_size=min_suffix_size,
                       suffix_data_split_value=suffix_data_split_value)


def read_split(spec : DatasetSpec,
               split : str,
               data_root : str | Path) -> pd.DataFrame:
    """
    Read one preprocessed split, keeping only the columns the spec declares.

    ARGS:
    - spec: The dataset specification.
    - split: One of 'train', 'val', 'test'.
    - data_root: The `data/` directory of the preprocessing repository.

    OUTPUTS:
    - df: One row per event, in the file's own order, with the time columns renamed.
    """
    assert split in SPLITS, f"Unknown split '{split}', expected one of {SPLITS}"

    reverse_renames = {new: old for old, new in COLUMN_RENAMES.items()}
    raw_continuous = [reverse_renames.get(col, col) for col in spec.continuous_columns]
    columns = [spec.case_name, *spec.categorical_columns, *raw_continuous, MIN_PREFIX_COLUMN]

    # Case ids and categorical values are read as strings: sepsis case ids are bare letters and
    # bpic19 carries boolean-looking attributes that pandas would otherwise infer as numbers.
    text_columns = dict.fromkeys([spec.case_name, *spec.categorical_columns], str)

    path = Path(data_root) / spec.name / 'processed' / f'{split}.csv'
    df = pd.read_csv(path, sep=CSV_SEPARATOR, usecols=columns, dtype=text_columns)
    df = df.rename(columns=COLUMN_RENAMES)

    for col in spec.continuous_columns:
        df[col] = df[col].astype('float32')

    return df


def _group_cases(df : pd.DataFrame,
                 case_name : str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Order the rows so that every case occupies one contiguous block, keeping the file's event order.

    A stable sort is used rather than an assumption of contiguity: the preprocessed logs are sorted
    by case start and then by timestamp, which interleaves two cases that start at the same instant.

    OUTPUTS:
    - order: Row positions of `df`, grouped by case.
    - case_ids: The case identifiers, one per case, in order of first appearance.
    - counts: Number of events per case, aligned with `case_ids`.
    """
    case_codes, case_ids = pd.factorize(df[case_name])
    order = np.argsort(case_codes, kind='stable')
    counts = np.bincount(case_codes, minlength=len(case_ids))
    return order, np.asarray(case_ids), counts


def add_eos_events(df : pd.DataFrame,
                   spec : DatasetSpec) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Append `min_suffix_size` end-of-sequence events to every case, grouping the rows by case.

    Mirrors `CSV2EventLog._CSV2EventLog__add_last_rows`: every categorical column of an EOS event
    holds the literal 'EOS', and every numerical column holds NaN, to be mean-imputed later.

    ARGS:
    - df: One split, one row per event, as returned by `read_split`.
    - spec: The dataset specification.

    OUTPUTS:
    - augmented: The split with the EOS events appended, one contiguous block per case.
    - case_ids: The case identifiers, one per case.
    - real_lengths: Number of real (non-EOS) events per case, aligned with `case_ids`.
    """
    order, case_ids, real_lengths = _group_cases(df, spec.case_name)
    ordered = df.iloc[order]

    augmented_lengths = real_lengths + spec.min_suffix_size
    augmented_offsets = np.concatenate(([0], np.cumsum(augmented_lengths)))

    # Row positions the real events take inside the augmented frame: every case's events land at the
    # front of its block, the appended EOS events at the back.
    real_offsets = np.concatenate(([0], np.cumsum(real_lengths)))
    within_case = np.arange(len(ordered)) - np.repeat(real_offsets[:-1], real_lengths)
    real_positions = np.repeat(augmented_offsets[:-1], real_lengths) + within_case

    is_real = np.zeros(augmented_offsets[-1], dtype=bool)
    is_real[real_positions] = True

    source = np.zeros(len(is_real), dtype=np.int64)
    source[real_positions] = np.arange(len(ordered))

    augmented = ordered.iloc[source].reset_index(drop=True)

    is_eos = ~is_real
    for col in spec.categorical_columns:
        augmented.loc[is_eos, col] = EOS_LABEL
    for col in spec.continuous_columns:
        augmented.loc[is_eos, col] = np.nan

    # Rewritten wholesale rather than patched on the EOS rows, since the copied source row carries
    # the wrong case for every appended event.
    case_per_row = np.repeat(np.arange(len(case_ids)), augmented_lengths)
    augmented[spec.case_name] = case_ids[case_per_row]

    min_prefix = df[MIN_PREFIX_COLUMN].to_numpy()[order][real_offsets[:-1]]
    augmented[MIN_PREFIX_COLUMN] = min_prefix[case_per_row]

    return augmented, case_ids, real_lengths


def _encode_columns(encoder_decoder : TensorEncoderDecoder,
                    spec : DatasetSpec,
                    df : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the fitted encoders to a whole split at once.

    `TensorEncoderDecoder.encode_categorical_column` and `encode_continuous_column` transform one
    case at a time; the transforms are row-independent, so transforming the split in one call gives
    the same values and stays linear in the number of events instead of the number of windows.

    OUTPUTS:
    - categorical: int32, [num_events, num_categorical_columns]. 0 is padding, unknown and missing.
    - continuous: float32, [num_events, num_continuous_columns]. Imputed and standardized.
    """
    categorical = np.zeros((len(df), len(spec.categorical_columns)), dtype=np.int32)
    for i, col in enumerate(spec.categorical_columns):
        values = df[[col]].astype(object).to_numpy()
        # +1 so that the encoder's unknown/missing value of -1 lands on 0, the padding index.
        categorical[:, i] = encoder_decoder.categorical_encoders[col].transform(values)[:, 0] + 1

    continuous = np.zeros((len(df), len(spec.continuous_columns)), dtype=np.float32)
    for i, col in enumerate(spec.continuous_columns):
        values = df[[col]].to_numpy()
        values = encoder_decoder.continuous_imputers[col].transform(values)
        continuous[:, i] = encoder_decoder.continuous_encoders[col].transform(values)[:, 0]

    return categorical, continuous


def _cut_points(spec : DatasetSpec,
                real_lengths : np.ndarray,
                min_prefix_lengths : np.ndarray,
                match_reference_cuts : bool) -> np.ndarray:
    """
    Enumerate the windows of every case.

    A window of `prefix_len` events is fed to the encoder minus its last `suffix_data_split_value`
    positions, so it conditions on the case's first `k = prefix_len - suffix_data_split_value`
    events. That `k` is the cut point the other model's `TraceDataset` enumerates.

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
                 spec : 'DatasetSpec',
                 categorical : np.ndarray,
                 continuous : np.ndarray,
                 offsets : np.ndarray,
                 case_ids : np.ndarray,
                 real_lengths : np.ndarray,
                 cuts : np.ndarray,
                 all_categories : tuple,
                 encoder_decoder : TensorEncoderDecoder,
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
        self.encoder_decoder = encoder_decoder
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

        Same rule as `TensorEncoderDecoder.pad_to_window_size`, applied to a whole window at once.
        Zero is the padding index for the categorical columns and, the columns being standardized,
        approximately the training mean for the continuous ones.
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


class PreSplitEventLogLoader:
    """
    Builds one `WindowDataset` per split from the preprocessed logs.

    Plays the role the original `EventLogLoader` used to play for a raw CSV, minus the feature
    engineering and the splitting, both of which the preprocessing pipeline already did.
    """

    def __init__(self,
                 spec : DatasetSpec,
                 data_root : str | Path):
        """
        ARGS:
        - spec: The dataset specification, normally from `spec_from_codec`.
        - data_root: The `data/` directory of the preprocessing repository.
        """
        self.spec = spec

        self.splits = {}
        for split in SPLITS:
            self.splits[split] = add_eos_events(read_split(spec, split, data_root), spec)

        train_df, _, _ = self.splits['train']

        # Fit on the training split only, end-of-sequence events included, exactly as the original
        # `EventLogLoader` used to: the imputers skip the NaNs those events carry, and the scalers
        # see them as the imputed mean.
        self.encoder_decoder = TensorEncoderDecoder(event_log=train_df, **spec.as_encoder_kwargs())
        self.encoder_decoder.train_imputers_encoders()

        # `TensorEncoderDecoder` keeps the frame it was fit on, and it travels with every dataset
        # that is pickled. Nothing reads it once the encoders are fit and the window size is known,
        # and on bpic17 it is a 700k row frame written out three times over.
        self.encoder_decoder.event_log = None

        self.all_categories = self._build_all_categories()

    def _build_all_categories(self) -> tuple[list, list]:
        """
        Feature descriptors in tensor order, as `TensorEncoderDecoder.encode_df` returns them.

        OUTPUTS:
        - all_categories: (categorical, numerical) lists of (column, number of labels, label map).
          The label map is 1-based, since 0 is the padding index.
        """
        categorical = []
        for col in self.spec.categorical_columns:
            classes = self.encoder_decoder.categorical_encoders[col].categories_[0]
            labels = {category: idx + 1 for idx, category in enumerate(classes)}
            categorical.append((col, len(classes) + 1, labels))

        numerical = [(col, 1, dict()) for col in self.spec.continuous_columns]

        return (categorical, numerical)

    def get_dataset(self, split : str) -> WindowDataset:
        """
        Encode one split and enumerate its windows.

        ARGS:
        - split: One of 'train', 'val', 'test'.

        OUTPUTS:
        - dataset: The split's windows.
        """
        df, case_ids, real_lengths = self.splits[split]
        categorical, continuous = _encode_columns(self.encoder_decoder, self.spec, df)

        augmented_lengths = real_lengths + self.spec.min_suffix_size
        offsets = np.concatenate(([0], np.cumsum(augmented_lengths)))[:-1]

        # One row per case, taken from the first event of each case.
        min_prefix_lengths = df[MIN_PREFIX_COLUMN].to_numpy()[offsets]

        # The test split is the one both models are scored on, so its windows are restricted to the
        # cut points the other model enumerates. The training and validation splits keep this
        # repository's own range; their bounds are 1 throughout, the out-of-time separation being
        # what raises them.
        cuts = _cut_points(self.spec,
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
                             encoder_decoder=self.encoder_decoder,
                             window_size=self.spec.window_size,
                             suffix_data_split_value=self.spec.suffix_data_split_value)
