"""
Encode the externally preprocessed splits into datasets this repository's trainer can read.
    python scripts/build_datasets.py --dataset sepsis bpic17 bpic19 --data-root <preprocessing>/data

Writes, per dataset and split, into `encoded_data/`:
- `<dataset>_all_<min_suffix_size>_<split>.pkl`, the dataset itself, in the naming the training
    use. It carries the codec it was encoded through with it, so the splits stay consistent.
- `<dataset>_all_<min_suffix_size>_<split>_index.csv`, one row per window, naming the case and the
  cut point it conditions on. The evaluation pipeline scores a generated suffix against the prefix
  it continues, and this is the only place that mapping exists.

The window geometry is not a flag: `configs/data.py` sets it, and the model is shaped from the same
numbers. `MIN_SUFFIX_SIZE` is part of the file names, so two datasets cut with different values can
sit side by side, but changing it is a change to that file rather than to a run.
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from configs.data import DATASETS, ENCODED_DIR, SEQ_LEN_PRED, encoded_stem  # noqa: E402
from configs.event_log import SPLITS  # noqa: E402
from event_log_loader import PreSplitEventLogLoader, spec_from_codec  # noqa: E402


def build(dataset : str,
          data_root : str):
    """
    Encode one dataset's three splits and write them out.

    ARGS:
    - dataset: Dataset name, i.e. the directory under `data_root`.
    - data_root: The `data/` directory of the preprocessing repository.
    """
    spec = spec_from_codec(dataset,
                           data_root=data_root,
                           suffix_data_split_value=SEQ_LEN_PRED)

    print(f"[{dataset}] window size: {spec.window_size}")
    print(f"[{dataset}] categorical features ({len(spec.categorical_columns)}): "
          f"{list(spec.categorical_columns)}")
    print(f"[{dataset}] numerical features ({len(spec.continuous_columns)}): "
          f"{list(spec.continuous_columns)}")

    loader = PreSplitEventLogLoader(spec, data_root=data_root)

    os.makedirs(ENCODED_DIR, exist_ok=True)
    for split in SPLITS:
        dataset_obj = loader.get_dataset(split)

        stem = f'{encoded_stem(dataset)}_{split}'
        torch.save(dataset_obj, f'{stem}.pkl')
        dataset_obj.index_frame().to_csv(f'{stem}_index.csv', index=False)

        print(f"[{dataset}] {split}: {len(dataset_obj)} windows over "
              f"{len(dataset_obj.case_ids)} cases -> {stem}.pkl")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', default=list(DATASETS), choices=DATASETS,
                        help='Datasets to encode.')
    parser.add_argument('--data-root', required=True,
                        help="The 'data/' directory of the preprocessing repository.")
    args = parser.parse_args()

    for dataset in args.dataset:
        build(dataset, data_root=args.data_root)


if __name__ == '__main__':
    main()
