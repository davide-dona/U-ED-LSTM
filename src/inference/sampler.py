import numpy as np
import torch
import torch.nn.functional as F

from configs.event_log import CASE_ELAPSED_COLUMN, EOS_LABEL, EVENT_ELAPSED_COLUMN
from configs.generation import GROWING_COLUMNS, POSITIVE_COLUMNS, SAMPLING

# Bounds the predicted log variances are read within. The training objective is defined on the log
# variance clamped to this range (`Loss.loss_attenuation_mse`), so sampling outside it would draw
# from a spread the model was never scored on, and a large enough value would overflow `exp`. They
# follow the loss, which is why they are not in `configs/generation.py` with the rest.
MIN_LOG_VARIANCE = -6.0
MAX_LOG_VARIANCE = 6.0


class SuffixSampler:
    """
    Suffix sampling for one model and one encoded split.

    ARGS:
    - model: A trained `DropoutUncertaintyEncoderDecoderLSTM`.
    - dataset: The `WindowDataset` the prefixes come from, naming the feature order and carrying the
      codec its values were encoded through.
    - device: Where the decoding runs.
    - use_variance_cat: Whether a categorical draw perturbs the logits by their predicted spread.
    - use_variance_num: Whether a numerical draw perturbs the mean by its predicted spread.
    - sample_argmax: Whether a stochastic categorical draw takes the argmax of the perturbed logits
      rather than sampling the softmax over them.
    - variational_dropout_sampling: Whether the decoder keeps one dropout mask for the whole suffix.
      False redraws it every step, which is the naive dropout the model is evaluated with.
    - positive_retries: Draws taken per step before a value that has to stay positive reads zero.
    - growing_columns: Numerical columns that may never decrease along a suffix.
    - positive_columns: Numerical columns that may never fall below zero.

    Every default is `configs/generation.py`'s, so that what a run drew with is configured in one
    place and can be stamped into the file it produced.
    """

    def __init__(self,
                 model,
                 dataset,
                 device : torch.device,
                 use_variance_cat : bool = SAMPLING['use_variance_cat'],
                 use_variance_num : bool = SAMPLING['use_variance_num'],
                 sample_argmax : bool = SAMPLING['sample_argmax'],
                 variational_dropout_sampling : bool = SAMPLING['variational_dropout_sampling'],
                 positive_retries : int = SAMPLING['positive_retries'],
                 growing_columns : tuple[str, ...] = GROWING_COLUMNS,
                 positive_columns : tuple[str, ...] = POSITIVE_COLUMNS):
        self.model = model.to(device)
        self.device = device

        self.use_variance_cat = use_variance_cat
        self.use_variance_num = use_variance_num
        self.sample_argmax = sample_argmax
        self.variational_dropout_sampling = variational_dropout_sampling
        self.positive_retries = positive_retries

        # The decoder's own features, in the order its input and output layers are built in.
        self.categorical_columns, self.numerical_columns = model.dec_feat
        # Where those features sit in the dataset's tensor lists, i.e. how a prefix is read. The
        # decoder embeds them in `dec_feat` order while reading them in dataset order, so the two
        # have to agree; they do for every spec `spec_from_codec` builds.
        categorical_indices, self.numerical_indices = model.data_indices_dec
        categorical_names, numerical_names = dataset.all_categories
        assert [categorical_names[i][0] for i in categorical_indices] \
            == list(self.categorical_columns), "Decoder categorical features are out of tensor order"
        assert [numerical_names[i][0] for i in self.numerical_indices] \
            == list(self.numerical_columns), "Decoder numerical features are out of tensor order"

        activity_column = dataset.spec.concept_name
        self.activity_slot = self.categorical_columns.index(activity_column)
        self.elapsed_slot = self.numerical_columns.index(CASE_ELAPSED_COLUMN)
        # The wait before an event, which the decoder predicts per step beside the activity. The
        # comparison reads it as the suffix's own timeline, one wait per generated event.
        self.wait_slot = self.numerical_columns.index(EVENT_ELAPSED_COLUMN)

        labels = next(labels for column, _, labels in categorical_names if column == activity_column)
        self.eos_index = labels[EOS_LABEL]

        self.growing = [column in growing_columns for column in self.numerical_columns]
        # The encoded value of a raw zero, which is what a positive column is sampled above.
        channels = dataset.codec.numeric
        self.positive_floor = [
            float(channels[column].normalize(np.zeros(1))[0]) if column in positive_columns else None
            for column in self.numerical_columns
        ]

        # A window reaches `min_suffix_size` events past the last real one, so a prefix cut after
        # `cut_point` events leaves this many steps before the window is exhausted.
        self.step_budget = dataset.window_size - dataset.spec.min_suffix_size

    def sample(self,
               prefix : list,
               cut_points : torch.Tensor,
               num_samples : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw `num_samples` stochastic suffixes for every prefix of a batch.

        ARGS:
        - prefix: The encoder input, `[categorical tensors, numerical tensors]`, each of shape
          [num_prefixes, window size - seq_len_pred].
        - cut_points: Number of real events each prefix holds: [num_prefixes].
        - num_samples: Suffixes drawn per prefix.

        OUTPUTS:
        - activities: Activity indices of every draw: [num_prefixes, num_samples, steps].
        - waits: Encoded `event_elapsed_time` of every step of every draw, i.e. the wait before the
          event at the same position: [num_prefixes, num_samples, steps].
        - lengths: Number of events of every draw: [num_prefixes, num_samples].
        - elapsed: Encoded `case_elapsed_time` of every draw's last event, or of the prefix's last
          event where the draw is empty: [num_prefixes, num_samples].
        """
        repeated = [[tensor.repeat_interleave(repeats=num_samples, dim=0) for tensor in tensors]
                    for tensors in prefix]
        repeated_cuts = cut_points.repeat_interleave(repeats=num_samples, dim=0)

        activities, waits, lengths, elapsed = self._decode(prefix=repeated,
                                                           cut_points=repeated_cuts,
                                                           stochastic=True)

        num_prefixes = cut_points.shape[0]
        return (activities.view(num_prefixes, num_samples, -1),
                waits.view(num_prefixes, num_samples, -1),
                lengths.view(num_prefixes, num_samples),
                elapsed.view(num_prefixes, num_samples))

    def point(self,
              prefix : list,
              cut_points : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                  torch.Tensor]:
        """
        Decode the deterministic suffix of every prefix of a batch.

        Every head is read at its mean, with dropout switched off for the duration, so the result is
        the single most likely continuation rather than a draw from the distribution around it.

        ARGS and OUTPUTS as `sample`, minus the sample dimension.
        """
        dropout_rates = self._disable_dropout()
        try:
            return self._decode(prefix=prefix, cut_points=cut_points, stochastic=False)
        finally:
            self._enable_dropout(dropout_rates)

    def _decode(self,
                prefix : list,
                cut_points : torch.Tensor,
                stochastic : bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
        """
        Free-run the decoder over a batch of prefixes until every row has ended.

        A row ends on the step it predicts the end-of-sequence activity, or once it has used up the
        window the encoders were fit for. Rows that end early keep being fed through the decoder, so
        the batch stays rectangular; nothing they produce afterwards is recorded.

        ARGS:
        - prefix: The encoder input, `[categorical tensors, numerical tensors]`.
        - cut_points: Number of real events each prefix holds.
        - stochastic: Whether the heads are sampled or read at their mean.

        OUTPUTS: as `sample`, without the sample dimension.
        """
        prefix = [[tensor.to(self.device) for tensor in tensors] for tensors in prefix]
        cut_points = cut_points.to(self.device)

        max_steps = self.step_budget - cut_points # [B]
        batch_size = cut_points.shape[0]

        predictions, (h, c), z = self.model.inference(prefix=prefix)

        # The last prefix event's numerical values, which the first predicted event is bounded by.
        last_numericals = [prefix[1][index][:, -1:] for index in self.numerical_indices]
        categoricals, numericals = self._draw(predictions, last_numericals, stochastic)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        elapsed = last_numericals[self.elapsed_slot].clone() # [B, 1]
        steps = []
        waits = []

        for step in range(int(max_steps.max().item()) + 1):
            activity = categoricals[self.activity_slot].squeeze(1) # [B]
            finished |= (activity == self.eos_index) | (step > max_steps)
            if bool(finished.all()):
                break

            active = ~finished
            steps.append(torch.where(active, activity, torch.zeros_like(activity)))
            # The wait is recorded with the event it leads up to - both come off the same draw -
            # and zeroed on a row that has already ended, exactly as its activity is: nothing
            # past a row's length is ever read.
            wait = numericals[self.wait_slot].squeeze(1) # [B]
            waits.append(torch.where(active, wait, torch.zeros_like(wait)))
            lengths += active.to(lengths.dtype)
            elapsed = torch.where(active.unsqueeze(1), numericals[self.elapsed_slot], elapsed)

            predictions, (h, c) = self.model.inference(
                last_event=(categoricals, numericals),
                hx=(h, c),
                z=z if self.variational_dropout_sampling else None,
            )
            categoricals, numericals = self._draw(predictions, numericals, stochastic)

        if steps:
            activities = torch.stack(tensors=steps, dim=1) # [B, steps]
            drawn_waits = torch.stack(tensors=waits, dim=1) # [B, steps]
        else:
            activities = torch.zeros((batch_size, 0), dtype=torch.int64, device=self.device)
            drawn_waits = torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)

        return activities, drawn_waits, lengths, elapsed.squeeze(1)

    def _draw(self,
              predictions : list,
              last_numericals : list[torch.Tensor],
              stochastic : bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Read one event off the decoder's heads, as the decoder's own next input.

        ARGS:
        - predictions: `[[categorical means, numerical means], [categorical log variances, numerical
          log variances]]`, as `DropoutUncertaintyEncoderDecoderLSTM.inference` returns them.
        - last_numericals: The previous event's numerical values, one [B, 1] tensor per decoder
          numerical feature, which the growing columns are bounded below by.
        - stochastic: Whether the heads are sampled or read at their mean.

        OUTPUTS:
        - categoricals: One [B, 1] index tensor per decoder categorical feature.
        - numericals: One [B, 1] value tensor per decoder numerical feature.
        """
        (categorical_means, numerical_means), (categorical_vars, numerical_vars) = predictions

        categoricals = []
        for column in self.categorical_columns:
            logits = categorical_means[f'{column}_mean']
            if not stochastic:
                categoricals.append(torch.argmax(logits, dim=-1, keepdim=True))
                continue

            if self.use_variance_cat:
                logvar = torch.clamp(categorical_vars[f'{column}_var'],
                                     min=MIN_LOG_VARIANCE,
                                     max=MAX_LOG_VARIANCE)
                logits = torch.normal(mean=logits, std=torch.exp(0.5 * logvar))

            if self.sample_argmax:
                categoricals.append(torch.argmax(logits, dim=-1, keepdim=True))
            else:
                probabilities = F.softmax(logits, dim=-1)
                categoricals.append(torch.multinomial(input=probabilities,
                                                      num_samples=1,
                                                      replacement=True))

        numericals = []
        for slot, column in enumerate(self.numerical_columns):
            mean = numerical_means[f'{column}_mean'] # [B, 1]

            if not stochastic or not self.use_variance_num:
                value = mean
            else:
                logvar = torch.clamp(numerical_vars[f'{column}_var'],
                                     min=MIN_LOG_VARIANCE,
                                     max=MAX_LOG_VARIANCE)
                std = torch.exp(0.5 * logvar)
                floor = self.positive_floor[slot]
                value = (torch.normal(mean=mean, std=std) if floor is None
                         else self._draw_above(mean=mean, std=std, floor=floor))

            if self.growing[slot]:
                value = torch.max(last_numericals[slot], value)

            numericals.append(value)

        return categoricals, numericals

    def _draw_above(self,
                    mean : torch.Tensor,
                    std : torch.Tensor,
                    floor : float) -> torch.Tensor:
        """
        Draw a value above `floor`, reading zero where none of the attempts clears it.

        ARGS:
        - mean, std: The distribution to draw from, [B, 1] each.
        - floor: The encoded value the draw has to exceed.

        OUTPUTS:
        - value: [B, 1].
        """
        candidates = torch.normal(mean=mean.expand(-1, self.positive_retries),
                                  std=std.expand(-1, self.positive_retries)) # [B, retries]
        above = candidates > floor
        # `argmax` reads the first `True`, and 0 both when the first candidate clears the floor and
        # when none does, hence the explicit fallback.
        first = torch.argmax(above.to(torch.int64), dim=1, keepdim=True)
        drawn = torch.gather(input=candidates, dim=1, index=first)

        return torch.where(above.any(dim=1, keepdim=True), drawn, torch.zeros_like(drawn))

    def _disable_dropout(self) -> tuple:
        """
        Switch dropout off throughout the model, returning the rates to restore it with.
        """
        model = self.model
        rates = (model.dropout,
                 model.encoder.first_layer.p_logit,
                 [layer.p_logit for layer in model.encoder.hidden_layers],
                 model.decoder.first_layer.p_logit,
                 [layer.p_logit for layer in model.decoder.hidden_layers])

        model.dropout = 0.0
        model.encoder.first_layer.p_logit = 0.0
        for layer in model.encoder.hidden_layers:
            layer.p_logit = 0.0
        model.decoder.first_layer.p_logit = 0.0
        for layer in model.decoder.hidden_layers:
            layer.p_logit = 0.0

        return rates

    def _enable_dropout(self, rates : tuple):
        """
        Restore the rates `_disable_dropout` returned.
        """
        model = self.model
        model.dropout = rates[0]
        model.encoder.first_layer.p_logit = rates[1]
        for layer, rate in zip(model.encoder.hidden_layers, rates[2]):
            layer.p_logit = rate
        model.decoder.first_layer.p_logit = rates[3]
        for layer, rate in zip(model.decoder.hidden_layers, rates[4]):
            layer.p_logit = rate
