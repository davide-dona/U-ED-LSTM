"""
What one dataset is, derived from the preprocessing pipeline's codec.

The codec is the single source of truth for the feature set and for the values every feature is
encoded through. This module reads the first half of it, the names: which columns exist and how the
windows over them are shaped. `encoding.py` reads the second half, the vocabularies and statistics.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from configs.data import MIN_SUFFIX_SIZE, SEQ_LEN_PRED
from configs.event_log import (
    CASE_COLUMN,
    CASE_ELAPSED_COLUMN,
    CASE_ELAPSED_KEY,
    EVENT_ELAPSED_COLUMN,
    REMAINING_TIME_COLUMN,
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


def read_codec(dataset : str,
               data_root : str | Path) -> dict:
    """
    Read one dataset's codec, as the preprocessing pipeline wrote it.

    ARGS:
    - dataset: Dataset name, i.e. the directory under `data_root`.
    - data_root: The `data/` directory of the preprocessing repository.

    OUTPUTS:
    - codec: The codec, as parsed JSON.
    """
    with open(Path(data_root) / dataset / 'codec' / 'dataset.json') as codec_file:
        return json.load(codec_file)


def spec_from_codec(dataset: str,
                    data_root : str | Path,
                    min_suffix_size : int = MIN_SUFFIX_SIZE,
                    suffix_data_split_value : int = SEQ_LEN_PRED) -> DatasetSpec:
    """
    Derive a `DatasetSpec` from the dataset codec of the preprocessing pipeline.

    The codec names the activity and resource columns and lists exactly the categorical and
    numerical attributes the other model reads, so driving the spec off it keeps the two models on
    the same features by construction.

    ARGS:
    - dataset: Dataset name, i.e. the directory under `data_root`.
    - data_root: The `data/` directory of the preprocessing repository.
    - min_suffix_size: Number of end-of-sequence events appended to every case.
    - suffix_data_split_value: Number of trailing window positions used as the decoder target.

    OUTPUTS:
    - spec: The dataset specification.
    """
    codec = read_codec(dataset, data_root)

    activity_column = codec['activity']['column']
    resource_column = codec['resource']['column']

    categorical_columns = (activity_column, resource_column) \
        + tuple(feature['column'] for feature in codec['categorical_features'])

    # `numeric_features` holds the time since case start beside the calendar features (the cyclical
    # day in week and seconds in day), which both models read as standardized numbers rather than as
    # categories. The time since case start is read as one of the two leading duration channels
    # instead, so it is filtered out here rather than listed a second time under the codec's name.
    numeric_columns = [feature['column'] for feature in codec['numeric_features']]
    assert CASE_ELAPSED_KEY in numeric_columns, \
        (f"'{dataset}' declares no '{CASE_ELAPSED_KEY}' event feature, which is the channel the "
         f"remaining runtime is read off. Add it to the dataset's `event_features` in the "
         f"preprocessing repository and preprocess it again")
    continuous_columns = (CASE_ELAPSED_COLUMN, EVENT_ELAPSED_COLUMN) \
        + tuple(col for col in numeric_columns if col != CASE_ELAPSED_KEY)

    duplicates = [col for col in set(categorical_columns) | set(continuous_columns)
                  if (categorical_columns + continuous_columns).count(col) > 1]
    assert not duplicates, f"Column(s) declared more than once by the codec: {duplicates}"
    assert REMAINING_TIME_COLUMN not in continuous_columns, \
        f"'{REMAINING_TIME_COLUMN}' is the prediction target and must not be an input feature"

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
