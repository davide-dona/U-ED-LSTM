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
numbers. `--min-suffix-size` is the exception, being part of the file names, so that two datasets
cut with different values can sit side by side.
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from configs.data import (  # noqa: E402
    DATASETS,
    ENCODED_DIR,
    MIN_SUFFIX_SIZE,
    SEQ_LEN_PRED,
    encoded_stem,
)
from configs.event_log import SPLITS  # noqa: E402
from event_log_loader import PreSplitEventLogLoader, spec_from_codec  # noqa: E402


def build(dataset : str,
          data_root : str,
          out_dir : str,
          min_suffix_size : int):
    """
    Encode one dataset's three splits and write them out.

    ARGS:
    - dataset: Dataset name, i.e. the directory under `data_root`.
    - data_root: The `data/` directory of the preprocessing repository.
    - out_dir: Where the datasets and their indices are written.
    - min_suffix_size: Number of end-of-sequence events appended to every case.
    """
    spec = spec_from_codec(dataset,
                           data_root=data_root,
                           min_suffix_size=min_suffix_size,
                           suffix_data_split_value=SEQ_LEN_PRED)

    print(f"[{dataset}] window size: {spec.window_size}")
    print(f"[{dataset}] categorical features ({len(spec.categorical_columns)}): "
          f"{list(spec.categorical_columns)}")
    print(f"[{dataset}] numerical features ({len(spec.continuous_columns)}): "
          f"{list(spec.continuous_columns)}")

    loader = PreSplitEventLogLoader(spec, data_root=data_root)

    os.makedirs(out_dir, exist_ok=True)
    for split in SPLITS:
        dataset_obj = loader.get_dataset(split)

        stem = f'{encoded_stem(out_dir, dataset, min_suffix_size)}_{split}'
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
    parser.add_argument('--out-dir', default=ENCODED_DIR,
                        help='Where the encoded datasets are written.')
    parser.add_argument('--min-suffix-size', type=int, default=MIN_SUFFIX_SIZE,
                        help='Number of end-of-sequence events appended to every case. '
                             f'Default: {MIN_SUFFIX_SIZE}.')
    args = parser.parse_args()

    for dataset in args.dataset:
        build(dataset,
              data_root=args.data_root,
              out_dir=args.out_dir,
              min_suffix_size=args.min_suffix_size)


if __name__ == '__main__':
    main()
