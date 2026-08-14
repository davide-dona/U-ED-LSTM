"""
The vocabularies and statistics every feature is encoded through, read off the codec.

The codec of the preprocessing pipeline already holds them: each categorical channel's vocabulary,
fit on the training split, and each numerical channel's training mean and deviation. They are read
from there rather than fit again here, for two reasons.

**Parity.** The two models being compared then standardize on identical statistics and share one
vocabulary, by construction rather than by coincidence. Fitting a second time is not free of
consequence: fitting a scaler *after* mean-imputation, as this repository used to, counts every
imputed row in the deviation while it contributes none, which shrinks it. On sepsis that put `Age`
at a deviation of 4.09 against the codec's 17.56 - the column is absent on seven rows in eight - so
the model saw it with roughly four times the spread the other model saw.

**One less fit to keep consistent.** Nothing is fit here at all, so there is no way for the encoders
travelling with a pickled dataset to disagree with the ones a later split was encoded through.

Index layout, per categorical channel: 0 is padding, 1 the end-of-sequence event, 2 an unknown
value, and the vocabulary follows from 3. Padding gets row 0 so that padding a short window is a
plain zero fill, and an unknown value gets a row of its own rather than sharing padding's.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from configs.event_log import COLUMN_RENAMES, EOS_LABEL, UNKNOWN_LABEL

from .spec import DatasetSpec, read_codec

# The rows every categorical channel carries before its vocabulary.
PAD_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2
FIRST_VOCAB_INDEX = 3

# The special rows, as the label map names them. Padding is absent: it stands for no value, and is
# the one row a decoded run can never legitimately hold.
SPECIAL_LABELS = {EOS_LABEL: EOS_INDEX, UNKNOWN_LABEL: UNK_INDEX}


@dataclass(frozen=True)
class CategoricalChannel:
    """
    One categorical feature, and the vocabulary it is embedded through.

    ATTRIBUTES:
    - column: The column of the split this channel reads.
    - vocab: The values the training split held, in the codec's own order.
    """

    column: str
    vocab: tuple[str, ...]

    def __post_init__(self):
        collisions = [label for label in SPECIAL_LABELS if label in self.vocab]
        assert not collisions, \
            (f"Column '{self.column}' holds {collisions} as ordinary value(s), which would collide "
             f"with the special row(s) of the same name")

    @property
    def num_rows(self) -> int:
        """Rows this channel owns, i.e. the size of its embedding."""
        return FIRST_VOCAB_INDEX + len(self.vocab)

    @cached_property
    def labels(self) -> dict[str, int]:
        """
        Every value this channel can hold, to the row standing for it.

        Doubles as the encoding map and as what a prediction is read back through: the two have to
        agree, so they are one dict.
        """
        return dict(SPECIAL_LABELS) \
            | {value: FIRST_VOCAB_INDEX + i for i, value in enumerate(self.vocab)}

    def encode(self, values : pd.Series) -> np.ndarray:
        """
        Map raw values to their rows, with an unknown row for what the training split did not hold.

        ARGS:
        - values: This channel's column of one split, as strings.

        OUTPUTS:
        - encoded: int32, [len(values)]. Missing values encode as unknown.
        """
        return values.map(self.labels).fillna(UNK_INDEX).to_numpy(dtype=np.int32)


@dataclass(frozen=True)
class NumericChannel:
    """
    One numerical feature, and the training statistics it is standardized against.

    ATTRIBUTES:
    - column: The column of the split this channel reads.
    - mean, std: Of the training split, the log transform included where there is one.
    - log: Whether the values pass through a log1p before being standardized.
    """

    column: str
    mean: float
    std: float
    log: bool

    @classmethod
    def from_codec(cls, entry : dict, column : str) -> 'NumericChannel':
        """
        ARGS:
        - entry: One numerical channel of the codec.
        - column: What this repository calls the column, which for the two durations is not what the
          codec calls it.
        """
        return cls(column=column, mean=entry['mean'], std=entry['std'], log=entry['log'])

    def normalize(self, values : np.ndarray) -> np.ndarray:
        """Standardize raw values against the training mean and deviation, keeping their shape."""
        values = np.asarray(values, dtype=np.float64)
        scaled = np.log1p(values) if self.log else values
        return (scaled - self.mean) / self.std

    def denormalize(self, values : np.ndarray) -> np.ndarray:
        """Read standardized values back as the quantity they came from, keeping their shape."""
        scaled = np.asarray(values, dtype=np.float64) * self.std + self.mean
        return np.expm1(scaled) if self.log else scaled

    def encode(self, values : pd.Series) -> np.ndarray:
        """
        Standardize this channel's column, pinning a missing value to exactly 0.0.

        The statistics were taken over the values the column does have, so a gap is zeroed rather
        than standardized: 0.0 is the training mean, which is what a mean-imputed gap encodes to.

        ARGS:
        - values: This channel's column of one split.

        OUTPUTS:
        - encoded: float32, [len(values)].
        """
        raw = values.to_numpy(dtype=np.float64)
        present = np.isfinite(raw)
        finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return (self.normalize(finite) * present).astype(np.float32)


class FeatureCodec:
    """
    Every channel of one dataset, in the spec's tensor order.

    Travels with each pickled dataset, so a dataset carries what it was encoded through and can read
    its own predictions back: `inference/generations.py` inverts the elapsed time channel with it,
    and `inference/sampler.py` reads the encoded value of a raw zero off it.
    """

    def __init__(self,
                 categorical : dict[str, CategoricalChannel],
                 numeric : dict[str, NumericChannel]):
        self.categorical = categorical
        self.numeric = numeric

    @classmethod
    def from_spec(cls,
                  spec : DatasetSpec,
                  data_root : str | Path) -> 'FeatureCodec':
        """
        Read the channels the spec declares out of the dataset's codec.

        ARGS:
        - spec: The dataset specification, whose column order the channels are kept in.
        - data_root: The `data/` directory of the preprocessing repository.

        OUTPUTS:
        - codec: The channels, one per feature of the spec.
        """
        codec = read_codec(spec.name, data_root)

        vocabularies = {entry['column']: entry['vocab']
                        for entry in [codec['activity'], codec['resource'],
                                      *codec['categorical_features']]}

        # The codec names the two durations `ts_start` and `ts_prev`; this repository renames them
        # on read, so their statistics are looked up under the codec's name and kept under ours.
        statistics = {COLUMN_RENAMES.get(entry['column'], entry['column']): entry
                      for entry in [codec['ts_start'], codec['ts_prev'],
                                    *codec['numeric_features']]}

        missing = [col for col in spec.categorical_columns if col not in vocabularies] \
            + [col for col in spec.continuous_columns if col not in statistics]
        assert not missing, f"The codec declares no channel for {missing}"

        return cls(
            categorical={col: CategoricalChannel(column=col, vocab=tuple(vocabularies[col]))
                         for col in spec.categorical_columns},
            numeric={col: NumericChannel.from_codec(statistics[col], column=col)
                     for col in spec.continuous_columns},
        )

    @property
    def all_categories(self) -> tuple[list, list]:
        """
        Feature descriptors in tensor order, which is what the model is shaped from.

        OUTPUTS:
        - all_categories: (categorical, numerical) lists of (column, number of rows, label map).
        """
        categorical = [(channel.column, channel.num_rows, channel.labels)
                       for channel in self.categorical.values()]
        numerical = [(channel.column, 1, dict()) for channel in self.numeric.values()]
        return (categorical, numerical)

    def encode(self, df : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a whole split at once.

        The transforms are row-independent, so the split goes through them in one call, which stays
        linear in the number of events rather than in the number of windows they are cut into.

        ARGS:
        - df: One split, one row per event, as `add_eos_events` returned it.

        OUTPUTS:
        - categorical: int32, [num_events, num_categorical_columns].
        - continuous: float32, [num_events, num_continuous_columns].
        """
        categorical = np.stack([channel.encode(df[channel.column])
                                for channel in self.categorical.values()], axis=1)
        continuous = np.stack([channel.encode(df[channel.column])
                               for channel in self.numeric.values()], axis=1)
        return categorical, continuous
