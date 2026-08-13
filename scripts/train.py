"""
Train the U-ED-LSTM on one of the encoded datasets.

    python scripts/build_datasets.py --dataset sepsis
    python scripts/train.py --dataset sepsis

Wires the model, the loss and the trainer together the same way the original training notebook
did, with the per-dataset hyperparameters in `configs/training.py`.

The encoder reads every feature the dataset carries; the decoder predicts the activity, the
resource and the two durations, as in the checkpoint shipped with this repository.
"""

import argparse
import os
import sys

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

from configs.training import DATASETS  # noqa: E402
from loss.losses import Loss  # noqa: E402
from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import (  # noqa: E402
    DropoutUncertaintyEncoderDecoderLSTM,
)
from trainer.trainer import Trainer  # noqa: E402

OPTIMIZERS = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}


class FlatSummaryWriter:
    """
    Wraps a SummaryWriter so the trainer's `add_scalars` calls land in a single event file.

    `SummaryWriter.add_scalars` opens a separate FileWriter (and subdirectory) per unique
    combination of tag keys, which fragments each run into dozens of event files. Trainer
    only ever calls `add_scalars`, so translating each call into one `add_scalar` per key,
    namespaced as "<main_tag>/<key>", keeps every scalar in the run's single event file
    while preserving the same grouping in the TensorBoard UI (it groups on '/').
    """

    def __init__(self, writer: SummaryWriter):
        self._writer = writer

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        for key, value in tag_scalar_dict.items():
            self._writer.add_scalar(f'{main_tag}/{key}', value, global_step)

    def close(self):
        self._writer.close()


def encoder_features(dataset) -> list[list[str]]:
    """
    Every feature of the dataset, in tensor order, as the encoder input list.

    OUTPUTS:
    - enc_feat: [categorical feature names, numerical feature names].
    """
    categorical, numerical = dataset.all_categories
    return [[feature[0] for feature in categorical], [feature[0] for feature in numerical]]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS),
                        help='Which encoded dataset to train on.')
    parser.add_argument('--encoded-dir', default='encoded_data',
                        help='Where `build_datasets.py` wrote the datasets.')
    parser.add_argument('--min-suffix-size', type=int, default=5,
                        help='The value the datasets were built with; part of their file names.')
    parser.add_argument('--out', default=None,
                        help='Checkpoint path. Defaults to checkpoints/<dataset>_u_ed_lstm.pkl.')
    parser.add_argument('--device', default=None,
                        help="Defaults to 'cuda' when available, otherwise 'cpu'.")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Overrides the dataset's configured number of epochs.")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Overrides the dataset's configured batch size.")
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Stop after this many epochs without validation improvement. '
                             'Unset disables early stopping.')
    args = parser.parse_args()

    config = DATASETS[args.dataset]

    stem = os.path.join(args.encoded_dir, f'{args.dataset}_all_{args.min_suffix_size}')
    data_train = torch.load(f'{stem}_train.pkl', weights_only=False)
    data_val = torch.load(f'{stem}_val.pkl', weights_only=False)
    print(f"Train windows: {len(data_train)}, validation windows: {len(data_val)}")

    enc_feat = encoder_features(data_train)
    dec_feat = data_train.spec.dec_feat
    print("Input features encoder: ", enc_feat)
    print("Features decoder: ", dec_feat)

    assert data_train.spec.suffix_data_split_value == config['seq_len_pred'], \
        (f"The dataset was built with a target of {data_train.spec.suffix_data_split_value} events "
         f"but the model predicts {config['seq_len_pred']}. Rebuild it with "
         f"--suffix-split {config['seq_len_pred']}.")

    model = DropoutUncertaintyEncoderDecoderLSTM(data_set_categories=data_train.all_categories,
                                                 enc_feat=enc_feat,
                                                 dec_feat=dec_feat,
                                                 seq_len_pred=config['seq_len_pred'],
                                                 hidden_size=config['hidden_size'],
                                                 num_layers=config['num_layers'],
                                                 dropout=config['dropout'])

    loss_obj = Loss()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    writer = FlatSummaryWriter(SummaryWriter(comment=f"_{args.dataset}_u_ed_lstm"))

    optimizer = OPTIMIZERS[config['optimizer']](params=model.parameters(),
                                                lr=config['learning_rate'],
                                                weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=config['scheduler_factor'],
                                  patience=config['scheduler_patience'],
                                  min_lr=config['scheduler_min_lr'])

    optimize_values = {
        'regularization_term': config['regularization_term'],
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epochs': args.epochs if args.epochs is not None else config['epochs'],
        'mini_batches': args.batch_size if args.batch_size is not None else config['batch_size'],
        'shuffle': config['shuffle'],
        'teacher_forcing_ratio': config['teacher_forcing_ratio'],
    }

    # GradNorm weights one loss per decoder output feature.
    gradnorm_values = {
        'number_tasks': len(dec_feat[0]) + len(dec_feat[1]),
        'gn_alpha': config['gn_alpha'],
        'gn_learning_rate': config['gn_learning_rate'],
    }

    out = args.out or os.path.join('checkpoints', f'{args.dataset}_u_ed_lstm.pkl')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    trainer = Trainer(device=device,
                      model=model,
                      data_train=data_train,
                      data_val=data_val,
                      loss_obj=loss_obj,
                      optimize_values=optimize_values,
                      suffix_data_split_value=data_train.spec.suffix_data_split_value,
                      writer=writer,
                      gradnorm_values=gradnorm_values,
                      saving_path=out,
                      early_stopping_patience=args.early_stopping_patience)

    trainer.train_model()
    writer.close()


if __name__ == '__main__':
    main()
