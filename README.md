# Probabilistic Suffix Prediction of Business Processes

## Probabilistic Suffix Prediction Framework
We predict a probability distribution of suffixes of business processes using our own U-ED-LSTM and MC Suffix Sampling Algorithm.

![Example Image](./img/example.png)

This repository is trimmed to the core U-ED-LSTM model code, kept for retraining and comparison against other models. Dataset loading/preprocessing, baseline model reimplementations, and evaluation/visualization are handled outside this repository.

## Setting Up the Python Environment with Pipenv

This project uses `pipenv` for managing Python dependencies. Follow the steps below to set up the virtual environment and install the necessary packages using the provided `Pipfile`.

### Prerequisites
Make sure you have Python and Pipenv installed.

### Setup Instructions

1. **Create the Virtual Environment**:

    ```bash
    pipenv install
    ```

2. **Activate the Virtual Environment**:

    ```bash
    pipenv shell
    ```

3. **Run the Project**: Inside the virtual environment, you have the Python packages installed for running the code.

## Repository Contents

- `src/model/dropout_uncertainty_enc_dec_LSTM/`: the U-ED-LSTM architecture (encoder, decoder, and dropout-uncertainty LSTM cell).
- `src/loss/losses.py`: the model's combined epistemic/aleatoric uncertainty loss.
- `src/trainer/trainer.py`: the GradNorm-based training loop.
- `src/event_log_loader/new_event_log_loader.py`: kept temporarily as a reference for the input tensor format the model expects (categorical/numerical feature encoding, `EventLogDataset`). It is not wired to any dataset-download or preprocessing pipeline here; a separate format adapter will translate our own precomputed, pre-split datasets into this model's expected input.
- `src/notebooks/training_variational_dropout/Helpdesk/full_enc_dec_lstm_gn.ipynb` (+ `Helpdesk_full_grad_norm_4layer.pkl`): a reference example showing how the model, loss, and trainer are wired together for training, along with the pretrained checkpoint used to validate this code reproduces the original paper's results.

> [!NOTE]
> The model/loss/trainer code here is pinned to the exact version that the pretrained checkpoints were trained with. A later revision of this repository changed the encoder/decoder architecture (added input projection + layer norm), which is **not** compatible with these checkpoints — verified by a strict `state_dict` load check.

## What's not here

Dataset download scripts, the full per-dataset loader/training/evaluation notebook pipeline, baseline model reimplementations (Camargo, Weytjens), and evaluation/visualization code have been removed. We use our own precomputed and pre-split datasets, and our own evaluation/visualization tooling (see `cvae-suffix-prediction`).
