"""
Reading one preprocessed split, and appending the end-of-sequence events every case is closed with.

Everything here works in the log's own values: the codec's vocabularies and statistics only come in
one stage later, in `encoding.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from configs.event_log import (
    COLUMN_RENAMES,
    CSV_SEPARATOR,
    EOS_LABEL,
    MIN_PREFIX_COLUMN,
    SPLITS,
)

from .spec import DatasetSpec


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
    holds the literal 'EOS', and every numerical column holds NaN, which the encoding pins to the
    training mean.

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
