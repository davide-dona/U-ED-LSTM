# Probabilistic Suffix Prediction of Business Processes

## Probabilistic Suffix Prediction Framework
We predict a probability distribution of suffixes of business processes using our own U-ED-LSTM and MC Suffix Sampling Algorithm.

![Example Image](./img/example.png)

This repository is trimmed to the core U-ED-LSTM model code, kept for retraining and comparison against other models. Dataset preprocessing, baseline model reimplementations, and evaluation/visualization are handled outside this repository, in `probabilistic-suffix-prediction`; an adapter here reads that repository's precomputed splits (see [Retraining on our datasets](#retraining-on-our-datasets)).

## Setting Up the Python Environment with uv

This project uses [`uv`](https://docs.astral.sh/uv/) for managing Python dependencies. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

### Prerequisites
Make sure you have [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed.

### Setup Instructions

1. **Sync the Virtual Environment**:

    ```bash
    uv sync
    ```

    This creates `.venv/` and installs the exact locked dependencies (add `--group dev` for `torch-tb-profiler`).

2. **Run the Project**: Prefix commands with `uv run`, e.g.

    ```bash
    uv run scripts/train.py --dataset sepsis
    ```

## Repository Contents

- `src/model/dropout_uncertainty_enc_dec_LSTM/`: the U-ED-LSTM architecture (encoder, decoder, and dropout-uncertainty LSTM cell).
- `src/loss/losses.py`: the model's combined epistemic/aleatoric uncertainty loss.
- `src/trainer/trainer.py`: the GradNorm-based training loop.
- `src/event_log_loader/new_event_log_loader.py`: `TensorEncoderDecoder`, which fits the imputers, ordinal encoders and scalers. The original `CSV2EventLog`, `EventLogSplitter`, `EventLogLoader` and `EventLogDataset` were removed: we bring our own preprocessing, split and dataset.
- `src/event_log_loader/presplit_loader.py`: the format adapter. Reads the precomputed `train.csv` / `val.csv` / `test.csv` from `probabilistic-suffix-prediction` and turns them into datasets the trainer consumes unchanged.
- `scripts/`: `build_datasets.py` (encode the splits), `train.py` (train one dataset).
- `configs/training.py`: per-dataset model and training hyperparameters, recovered from the pre-strip history.
- `configs/event_log.py`: constants for the event log schema written by the external preprocessing pipeline.
- `src/notebooks/training_variational_dropout/Helpdesk/Helpdesk_full_grad_norm_4layer.pkl`: the pretrained checkpoint used to validate this code reproduces the original paper's results. `scripts/train.py` is now the only training entry point, wired the same way this checkpoint was originally trained.

> [!NOTE]
> The model/loss/trainer code here is pinned to the exact version that the pretrained checkpoints were trained with. A later revision of this repository changed the encoder/decoder architecture (added input projection + layer norm), which is **not** compatible with these checkpoints — verified by a strict `state_dict` load check.

## Retraining on our datasets

Supported datasets: **sepsis**, **bpic17**, **bpic19**. All three are driven off
`data/<dataset>/codec/dataset.json` in `probabilistic-suffix-prediction`, which is the single source
of truth for the feature set, so both models read exactly the same features.

```bash
# Encode the precomputed splits into encoded_data/
uv run scripts/build_datasets.py --dataset sepsis bpic17 bpic19 \
    --data-root ~/GitHub/probabilistic-suffix-prediction/data

# Train
uv run scripts/train.py --dataset sepsis
```

What the adapter guarantees:

- **Same features.** Activity, resource, the two durations, and exactly the categorical and numerical
  attributes the codec declares. The remaining time is the prediction target and is never read as an
  input.
- **Same split.** Our out-of-time split is used as given. The original random `EventLogSplitter` is
  not.
- **Same test population.** A window of `prefix_len` events conditions the encoder on the case's
  first `prefix_len - seq_len_pred` events, so it *is* a cut point. The test windows are exactly the
  cut points the other model is scored on, `min_prefix_len` bounds included, verified pair for pair.
- **Traceability.** `<dataset>_all_5_<split>_index.csv` maps every window back to its
  `(case_id, cut_point)`, which is what the generations file will need.

Two things to know before a long run:

- `Trainer` is GradNorm-only. The alternative path it used to carry never backpropagated the
  numerical losses, so it was removed rather than left as a trap; `gradnorm_values` is now a
  required argument.
- The windows are built on access rather than up front. Materializing them eagerly, as
  `TensorEncoderDecoder.encode_df` does, costs about 7 GB of tensors on bpic17.

## What's not here

Dataset download scripts, the full per-dataset loader/training/evaluation notebook pipeline, baseline model reimplementations (Camargo, Weytjens), and evaluation/visualization code have been removed. We use our own precomputed and pre-split datasets, and our own evaluation/visualization tooling (see `probabilistic-suffix-prediction`).

Still to build: the output adapter. Sampling suffixes with `model.inference` and writing them into
the generations Parquet that `pipelines/evaluate.py` scores is not implemented here yet.
