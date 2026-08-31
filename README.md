# Probabilistic Suffix Prediction of Business Processes

## Probabilistic Suffix Prediction Framework
We predict a probability distribution of suffixes of business processes using our own U-ED-LSTM and MC Suffix Sampling Algorithm.

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
- `src/event_log_loader/`: the read side of the format adapter, one module per stage. It reads the precomputed `train.csv` / `val.csv` / `test.csv` from `probabilistic-suffix-prediction` and turns them into datasets the trainer consumes unchanged. The original `CSV2EventLog`, `EventLogSplitter`, `EventLogLoader`, `EventLogDataset` and `TensorEncoderDecoder` were removed: we bring our own preprocessing, split, dataset and encoding.
  - `spec.py`: which features exist and how the windows are shaped, off the preprocessing codec.
  - `reader.py`: one split's own values, with the end-of-sequence events appended to every case.
  - `encoding.py`: the vocabularies and statistics the features are encoded through, off the codec.
  - `windows.py`: the prefix windows, and the `Dataset` serving them.
  - `loader.py`: the four of them wired together.
- `src/inference/sampler.py`: MC suffix sampling. Draws stochastic suffixes off `model.inference`, and the deterministic point prediction beside them. Records the activity and the wait of every step, both off the same draw.
- `src/inference/generations.py`: the write side of the format adapter. Decodes what the sampler produced back into the log's alphabet and minutes, and writes the generations Parquet the comparison is scored from.
- `scripts/`: `build_datasets.py` (encode the splits), `train.py` (train one dataset), `generate.py` (sample test set suffixes).
- `configs/data.py`: which datasets exist, and the window geometry every script and the model are built from.
- `configs/training.py`: per-dataset model and training hyperparameters, recovered from the pre-strip history.
- `configs/generation.py`: what a generations run draws with.
- `configs/event_log.py`: constants for the event log schema written by the external preprocessing pipeline.

> [!NOTE]
> The model/loss/trainer code here is pinned to the exact version that the pretrained checkpoints were trained with. A later revision of this repository changed the encoder/decoder architecture (added input projection + layer norm), which is **not** compatible with these checkpoints — verified by a strict `state_dict` load check.

> [!NOTE]
> `DropoutUncertaintyLSTMCell` holds the four gates as one stacked parameter per projection instead
> of eight separate `nn.Linear` modules. This is a throughput change only: the cell computes the
> same function, with the same per-gate dropout masks, the same two bias vectors per gate and the
> same regularizer, and it still loads the pretrained checkpoints (the eight-Linear layout is
> remapped on load). It is the parameter *layout* that changed, not the architecture. What it buys
> is roughly five times fewer matmul kernels per training step, each four times wider, which is what
> the model was actually bottlenecked on.

## Retraining on our datasets

Supported datasets: **sepsis**, **bpic17**, **bpic17-dr**, **bpic19**. All four are driven off
`data/<dataset>/codec/dataset.json` in `probabilistic-suffix-prediction`, which is the single source
of truth for the feature set, so both models read exactly the same features.

```bash
# Encode the precomputed splits into encoded_data/
uv run scripts/build_datasets.py --dataset sepsis bpic17 bpic17-dr bpic19 \
    --data-root ~/GitHub/probabilistic-suffix-prediction/data

# Train
uv run scripts/train.py --dataset sepsis

# Sample 100 test set suffixes per prefix, as the comparison's generations file
uv run scripts/generate.py --dataset sepsis
```

What the adapter guarantees:

- **Same features.** Activity, resource, the two durations, and exactly the categorical and numerical
  attributes the codec declares. The remaining time is the prediction target and is never read as an
  input. The codec names the two durations in two places - the time till next event under
  `time_to_next`, the time since case start among the event features, since that is what
  preprocessing derives it as - and the second is read as a duration channel rather than a second
  time as an attribute.
- **Same encoding.** The codec's vocabularies and its training mean and deviation are read as given,
  not fit a second time here, so both models standardize on identical statistics and share one
  vocabulary. Nothing is fit in this repository at all.
- **Same split.** Our out-of-time split is used as given. The original random `EventLogSplitter` is
  not.
- **Same test population.** A window of `prefix_len` events conditions the encoder on the case's
  first `prefix_len - seq_len_pred` events, so it *is* a cut point. The test windows are exactly the
  cut points the other model is scored on, `min_prefix_len` bounds included, verified pair for pair.
- **Traceability.** `<dataset>_all_5_<split>_index.csv` maps every window back to its
  `(case_id, cut_point)`, which is what the generations file will need.

Three things to know before a long run:

- Every setting lives in `configs/`, not in the scripts. The scripts take only what varies between
  two runs on the same machine: `--dataset` and `--device`, plus `--data-root` when encoding. An
  experiment is changed by editing a config file, so a run is reproducible from the tree it was
  launched from.
- The window geometry in particular lives in `configs/data.py`: `MIN_SUFFIX_SIZE` (the
  end-of-sequence events appended to every case) and `SEQ_LEN_PRED` (the trailing window positions
  the decoder predicts, which is the loader's `suffix_data_split_value`). Changing either
  invalidates the encoded datasets and every checkpoint trained on them.
- `Trainer` is GradNorm-only. The alternative path it used to carry never backpropagated the
  numerical losses, so it was removed rather than left as a trap; `gradnorm_values` is now a
  required argument.
- The GradNorm loss updates the task weights and nothing else. It used to be backpropagated with
  `loss.backward()`, which also deposited a gradient on every model parameter, and that is a term
  Chen et al. exclude. Taking each gradient norm on the task's own loss and multiplying the weight
  back in afterwards leaves the weight update identical (`||grad(w_i L_i)||` is `|w_i|
  ||grad(L_i)||`) while removing both the stray gradient and a double backward over the whole
  recurrence, which was 40% of a training step. Runs from before this change are not comparable
  with runs after it.
- The windows are built on access rather than up front. Materializing them eagerly would cost about
  7 GB of tensors on bpic17, against 50 MB for the events they are cut from.

## Generating suffixes

`scripts/generate.py` writes `outputs/generations/<dataset>/<model>/<tag>.parquet`, the file the
comparison scores every model from. It holds one row per test prefix, with 100 sampled activity
suffixes, their per-event waits and their remaining runtimes nested inside it, the deterministic
point prediction, and the ground truth. The run identity (`dataset`, `model`, `tag`) is stamped into the file's schema
metadata, so it still says what produced it once it has been moved next to the other models' results.

- **It says how it was drawn.** Beside the identity, the file carries the settings the suffixes were
  drawn with: the checkpoint, the seed, the dropout rate, and every entry of `configs/generation.py`'s
  `SAMPLING` as the run resolved it. One dict both builds the sampler and is stamped into the file,
  so what a file says it was drawn with is what it was drawn with.
- **A hundred samples.** `NUM_SAMPLES` in `configs/generation.py` is 100, what the comparison's
  distribution metrics are read off and what the other two models draw: the CVAE's
  `inference.evaluation_samples` and SuTraN's own `NUM_SAMPLES`. Runs before 2026-08-30 drew 10,
  the smallest number hit-rate-at-10 can be read from.
- **One wait per event.** Beside each activity suffix the file carries
  `generated_time_to_next_minutes`, and the same for the point prediction and the ground truth.
  A wait list runs exactly as far as the activity list beside it, so the two are read position for
  position: the wait at position `d` is the gap between the events at `d-1` and `d`, measured from
  the last prefix event at `d = 0`. That is the `event_elapsed_time` the decoder already predicts at
  every step, which is why this costs no head and no retraining of its own. It is *not* constrained
  against the remaining runtime below - the two come off different channels - so a draw's waits do
  not sum to its remaining runtime. SuTraN has the same split, and the comparison scores the two
  separately.
- **Every draw is independent.** The dropout masks are drawn per batch row, so repeating a prefix
  along the batch gives each of its 100 draws its own encoder and decoder dropout. All draws of a
  batch of prefixes therefore decode as one tensor batch, which is what makes bpic17's 250k prefixes
  tractable - and why `BATCH_PREFIXES` is 128 rather than the 1024 that was right at 10 draws.
- **Remaining time.** The model has no remaining runtime head. It predicts `case_elapsed_time` per
  event and is trained to keep it monotone, so a suffix's remaining runtime is its last event's
  `case_elapsed_time` less the prefix's own. The ground truth is read the same way off the encoded
  split, which makes it the log's `rtime` column by identity.

Score it from the other repository:

```bash
uv run python -m pipelines.evaluate \
    -g <this repo>/outputs/generations/sepsis/u-ed-lstm/<tag>.parquet
```

## What's not here

Dataset download scripts, the full per-dataset loader/training/evaluation notebook pipeline, baseline model reimplementations (Camargo, Weytjens), and evaluation/visualization code have been removed. We use our own precomputed and pre-split datasets, and our own evaluation/visualization tooling (see `probabilistic-suffix-prediction`).
