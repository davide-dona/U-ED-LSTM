"""
Sample test set suffixes with a trained U-ED-LSTM and write them as the comparison's generations file.

    python scripts/build_datasets.py --dataset sepsis --data-root <preprocessing repo>/data
    python scripts/train.py --dataset sepsis
    python scripts/generate.py --dataset sepsis --model checkpoints/sepsis_u_ed_lstm.pkl

Draws `--num-samples` suffixes per test prefix by MC suffix sampling, and the deterministic point
prediction beside them. No metrics are computed here: the predictions are decoded back into the log's
own alphabet and minutes and written as the Parquet the comparison scores every model from, which
`pipelines/evaluate.py` of `probabilistic-suffix-prediction` reads.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from configs.data import DATASETS, ENCODED_DIR, MIN_SUFFIX_SIZE, encoded_stem  # noqa: E402
from configs.generation import BATCH_PREFIXES, NUM_SAMPLES, SAMPLING  # noqa: E402
from inference.generations import (  # noqa: E402
    MODEL_NAME,
    GenerationDecoder,
    RunIdentity,
    open_generations,
    table_from_rows,
)
from inference.sampler import SuffixSampler  # noqa: E402
from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import (  # noqa: E402
    DropoutUncertaintyEncoderDecoderLSTM,
)


def run_tag() -> str:
    """A tag telling two runs of this model on one dataset apart."""
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def generations_path(run : RunIdentity) -> str:
    """
    Where one run's generations file is written.

    Mirrors the layout of the comparison's own results (`src/paths.py` of the
    `probabilistic-suffix-prediction` repository), with the model above the tag so one log's directory
    lists the models compared on it before any filename is read. The path is only a mirror: the run
    identity is stamped inside the file, so it can be moved next to the other models' results and
    still say what produced it.
    """
    return os.path.join('outputs', 'generations', run.dataset, run.model, f'{run.tag}.parquet')


def generation_order(dataset) -> np.ndarray:
    """
    The dataset rows to generate for, ordered so that a batch decodes roughly uniformly.

    Sorted by how many events actually follow the cut. A batch runs until its last row has ended, so
    mixing a two-event suffix with a fifty-event one would decode the short one forty-eight times
    over for nothing.
    """
    index = dataset.index_frame()
    return index.sort_values(by='suffix_len', kind='stable')['row'].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS),
                        help='Which encoded dataset to generate for.')
    parser.add_argument('--model', required=True,
                        help='Checkpoint to generate with, as written by scripts/train.py.')
    parser.add_argument('--encoded-dir', default=ENCODED_DIR,
                        help='Where `build_datasets.py` wrote the datasets.')
    parser.add_argument('--min-suffix-size', type=int, default=MIN_SUFFIX_SIZE,
                        help='The value the datasets were built with; part of their file names.')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES,
                        help='Suffixes drawn per test prefix, beside the point prediction. '
                             f'Default: {NUM_SAMPLES}.')
    parser.add_argument('--batch-prefixes', type=int, default=BATCH_PREFIXES,
                        help='Prefixes decoded at once, and hence rows per Parquet row group. '
                             f'Default: {BATCH_PREFIXES}.')
    parser.add_argument('--dropout', type=float, default=None,
                        help="Dropout rate to sample with. Defaults to the checkpoint's own.")

    sampling_group = parser.add_argument_group(
        'sampling',
        'What a draw is, as against how fast it is produced. Defaults from configs/generation.py, '
        'and whatever they resolve to is stamped into the file that is written.')
    sampling_group.add_argument('--variance-cat', action=argparse.BooleanOptionalAction,
                          default=SAMPLING['use_variance_cat'],
                          help='Perturb the logits of a draw by their predicted spread.')
    sampling_group.add_argument('--variance-num', action=argparse.BooleanOptionalAction,
                          default=SAMPLING['use_variance_num'],
                          help='Perturb a numerical mean by its predicted spread.')
    sampling_group.add_argument('--sample-argmax', action=argparse.BooleanOptionalAction,
                          default=SAMPLING['sample_argmax'],
                          help='Take the argmax of the perturbed logits rather than sampling them.')
    sampling_group.add_argument('--variational-dropout', action=argparse.BooleanOptionalAction,
                          default=SAMPLING['variational_dropout_sampling'],
                          help='Keep one decoder dropout mask for a whole suffix.')
    sampling_group.add_argument('--positive-retries', type=int,
                          default=SAMPLING['positive_retries'],
                          help='Draws per step before a positive value gives up and reads zero.')
    parser.add_argument('--device', default=None,
                        help="Defaults to 'cuda' when available, otherwise 'cpu'.")
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed making the sampled suffixes reproducible.')
    parser.add_argument('--model-name', default=MODEL_NAME,
                        help=f"Name this run is compared under. Default: '{MODEL_NAME}'.")
    parser.add_argument('--tag', default=None,
                        help='What tells two runs of this model on this dataset apart. Default: the '
                             'current time, as %%Y%%m%%d-%%H%%M%%S.')
    parser.add_argument('--out', default=None,
                        help='Parquet file to write. Defaults to '
                             'outputs/generations/<dataset>/<model>/<tag>.parquet.')
    args = parser.parse_args()

    assert args.num_samples >= 1, \
        ('A generations file holds the suffixes drawn per prefix, so --num-samples must be at least '
         f'1, got {args.num_samples}.')

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run = RunIdentity(dataset=args.dataset, model=args.model_name, tag=args.tag or run_tag())
    path = args.out or generations_path(run)

    stem = encoded_stem(args.encoded_dir, args.dataset, args.min_suffix_size)
    dataset = torch.load(f'{stem}_test.pkl', weights_only=False)

    model = DropoutUncertaintyEncoderDecoderLSTM.load(args.model, dropout=args.dropout)
    model.eval()

    # One dict, used to build the sampler and stamped into the file, so that what a generations file
    # says it was drawn with is what it was drawn with.
    sampling = {
        'use_variance_cat': args.variance_cat,
        'use_variance_num': args.variance_num,
        'sample_argmax': args.sample_argmax,
        'variational_dropout_sampling': args.variational_dropout,
        'positive_retries': args.positive_retries,
    }
    settings = sampling | {
        'num_samples': args.num_samples,
        'dropout': float(model.dropout),
        'checkpoint': args.model,
        'seed': args.seed,
    }

    sampler = SuffixSampler(model=model, dataset=dataset, device=device, **sampling)
    decoder = GenerationDecoder(dataset)

    order = generation_order(dataset)
    loader = DataLoader(dataset=Subset(dataset, order.tolist()),
                        batch_size=args.batch_prefixes,
                        shuffle=False)
    cut_points = torch.from_numpy(dataset.cuts[order, 1] - dataset.suffix_data_split_value)

    split = dataset.suffix_data_split_value
    written = 0
    with open_generations(path=path, run=run, settings=settings) as writer:
        for categoricals, numericals, _ in tqdm(loader, desc='Generating', unit='batch'):
            # The encoder reads the window minus the positions the decoder is trained to predict,
            # exactly as the trainer slices it.
            prefix = [[tensor[:, :-split] for tensor in categoricals],
                      [tensor[:, :-split] for tensor in numericals]]

            # The loader keeps `order`, so a batch is the next contiguous run of it.
            end = written + categoricals[0].shape[0]
            batch_cuts = cut_points[written:end]

            rows = decoder.rows(
                row_indices=order[written:end],
                sampled=sampler.sample(prefix=prefix,
                                       cut_points=batch_cuts,
                                       num_samples=args.num_samples),
                point=sampler.point(prefix=prefix, cut_points=batch_cuts),
            )
            writer.write_table(table=table_from_rows(rows))
            written = end

    print(f'Wrote {written} prefixes of {run} to {path}')


if __name__ == '__main__':
    main()
