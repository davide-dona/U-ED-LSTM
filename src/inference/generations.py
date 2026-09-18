import json
import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from configs.event_log import CASE_ELAPSED_COLUMN, EVENT_ELAPSED_COLUMN, UNKNOWN_LABEL

# Metadata keys required by suffix-generation's generation-store reader.
PROVENANCE_KEY = b'provenance'
VOCABULARY_KEY = b'activities'

# What this fork calls itself in a run identity, and hence in the comparison's tables.
MODEL_NAME = 'u_ed_lstm'

# One run's wait until each of its activities. Timestamps are these accumulated, so they are not
# written a second time.
_INTER_EVENT_TIMES = pa.list_(pa.field(name='element', type=pa.float32()))

_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('prefix_activities', pa.string()),
        ('generated_suffixes', pa.list_(pa.field(name='element', type=pa.string()))),
        ('generated_draws', pa.list_(pa.field(name='element', type=pa.int16()))),
        ('generated_inter_event_time_minutes',
         pa.list_(pa.field(name='element', type=_INTER_EVENT_TIMES))),
        ('generated_remaining_time_minutes', pa.list_(pa.field(name='element', type=pa.float32()))),
        ('point_activities', pa.string()),
        ('point_inter_event_time_minutes', _INTER_EVENT_TIMES),
        ('point_remaining_time_minutes', pa.float32()),
        ('true_activities', pa.string()),
        ('true_inter_event_time_minutes', _INTER_EVENT_TIMES),
        ('true_remaining_time_minutes', pa.float32()),
    ]
)


@dataclass(frozen=True)
class RunIdentity:
    """
    One model's one run on one dataset, stamped into the file it produced.

    Carried inside the generations file rather than spelled into its path, so it still says what it
    is once it has been moved next to the other models' results.

    ATTRIBUTES:
    - dataset: Name of the dataset generated for.
    - model: Name this run is compared under.
    - run_id: What tells two runs of one model on one dataset apart, formatted as
      `%Y%m%d-%H%M%S-%f`.
    """

    dataset: str
    model: str
    run_id: str

    def __str__(self):
        """What a message calls this run, e.g. `sepsis/u_ed_lstm/20260813-142910-123456`."""
        return f'{self.dataset}/{self.model}/{self.run_id}'


def open_generations(path : str,
                     run : RunIdentity,
                     checkpoint_sha256 : str,
                     vocabulary : list[str]) -> pq.ParquetWriter:
    """
    Open a generations file for writing, creating the directories it sits in.

    ARGS:
    - path: The Parquet file to write. An existing file is overwritten.
    - run: The run whose identity is stamped into the file's schema metadata.
    - checkpoint_sha256: SHA-256 digest of the checkpoint that produced the file.
    - vocabulary: Activity labels in the code order used by the string columns.

    OUTPUTS:
    - writer: To be closed, or used as a context manager, by the caller.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    provenance = {
        'dataset': run.dataset,
        'model': run.model,
        'run_id': run.run_id,
        'checkpoint_sha256': checkpoint_sha256,
    }
    schema = _SCHEMA.with_metadata({PROVENANCE_KEY: json.dumps(provenance).encode(),
                                    VOCABULARY_KEY: json.dumps(vocabulary).encode()})
    return pq.ParquetWriter(where=path, schema=schema)


def table_from_rows(rows : list[dict]) -> pa.Table:
    """
    One batch of prefixes as a table, and hence as one row group of the generations file.

    The evaluation pipeline hands one row group to one worker process, so the batch size is the
    granularity its parallelism is bounded by.
    """
    return pa.Table.from_pylist(mapping=rows, schema=_SCHEMA)


class GenerationDecoder:
    """
    Reads the decoder's output, and the split's own ground truth, back into the log's alphabet.

    ARGS:
    - dataset: The `WindowDataset` the prefixes were taken from.
    """

    def __init__(self, dataset):
        self.dataset = dataset

        activity_column = dataset.spec.concept_name
        self.activity_channel = dataset.spec.categorical_columns.index(activity_column)
        self.elapsed_channel = dataset.spec.continuous_columns.index(CASE_ELAPSED_COLUMN)
        self.elapsed = dataset.codec.numeric[CASE_ELAPSED_COLUMN]
        # The wait before an event. The remaining runtime is read off the elapsed time above, one
        # value per prefix; this is read per event, so a draw carries its own timeline.
        self.wait_channel = dataset.spec.continuous_columns.index(EVENT_ELAPSED_COLUMN)
        self.wait = dataset.codec.numeric[EVENT_ELAPSED_COLUMN]

        labels = next(labels for column, _, labels in dataset.all_categories[0]
                      if column == activity_column)
        # Row 0 is padding, and it is the one row `labels` does not name. Nothing in a decoded run is
        # padding - the prefix and the ground truth are read off the events themselves, and a
        # generated run stops before the end-of-sequence row - so the fill is never reached.
        self.names = [UNKNOWN_LABEL] * (max(labels.values()) + 1)
        for label, index in labels.items():
            self.names[index] = label
        # suffix-generation stores activity strings as private-use characters and carries the
        # corresponding labels in its schema metadata. Include every canonical special label even
        # though this model cannot emit padding or start-of-suffix events.
        self.vocabulary = ['PAD', 'EOT', 'SOS', UNKNOWN_LABEL,
                           *[label for label, _ in sorted(labels.items(), key=lambda item: item[1])
                             if label not in {'EOT', UNKNOWN_LABEL}]]
        self.codes = {label: chr(0xE000 + index)
                      for index, label in enumerate(self.vocabulary)}

    def rows(self,
             row_indices : np.ndarray,
             sampled : tuple,
             point : tuple) -> list[dict]:
        """
        Assemble one batch of prefixes into generations file rows.

        ARGS:
        - row_indices: The dataset rows this batch decoded, in batch order: [num_prefixes].
        - sampled: `(activities, waits, lengths, elapsed)` as `SuffixSampler.sample` returned them.
        - point: `(activities, waits, lengths, elapsed)` as `SuffixSampler.point` returned them.

        OUTPUTS:
        - rows: One dict per prefix, matching the generations file's schema.
        """
        dataset = self.dataset
        case_indices, window_lengths = dataset.cuts[row_indices, 0], dataset.cuts[row_indices, 1]
        cut_points = window_lengths - dataset.suffix_data_split_value
        starts = dataset.offsets[case_indices]
        real_lengths = dataset.real_lengths[case_indices]

        # The prefix's own elapsed time is what every remaining runtime of the row is measured from.
        prefix_elapsed = self._minutes(self.elapsed,
                                       dataset.continuous[starts + cut_points - 1,
                                                          self.elapsed_channel])
        true_elapsed = self._minutes(self.elapsed,
                                     dataset.continuous[starts + real_lengths - 1,
                                                        self.elapsed_channel])

        sampled_activities, sampled_waits, sampled_lengths, sampled_elapsed = (
            tensor.cpu().numpy() for tensor in sampled)
        point_activities, point_waits, point_lengths, point_elapsed = (tensor.cpu().numpy()
                                                                       for tensor in point)
        sampled_remaining = self._remaining(self._minutes(self.elapsed, sampled_elapsed),
                                            prefix_elapsed[:, None])
        point_remaining = self._remaining(self._minutes(self.elapsed, point_elapsed),
                                          prefix_elapsed)

        # De-standardized a batch at a time rather than over the whole split, which keeps this
        # bounded by one batch of prefixes like everything else here.
        sampled_minutes = self._waits(sampled_waits)
        point_minutes = self._waits(point_waits)

        events = dataset.categorical[:, self.activity_channel]
        true_waits = dataset.continuous[:, self.wait_channel]

        rows = []
        for row in range(len(row_indices)):
            start, cut_point = starts[row], cut_points[row]

            rows.append({
                'case_id': str(dataset.case_ids[case_indices[row]]),
                'prefix_len': int(cut_point),
                'prefix_activities': self._code(events[start : start + cut_point]),
                **self._draws(sampled_activities[row], sampled_lengths[row],
                              sampled_minutes[row], sampled_remaining[row]),
                'point_activities': self._code(point_activities[row, :point_lengths[row]]),
                'point_inter_event_time_minutes': point_minutes[row, :point_lengths[row]].tolist(),
                'point_remaining_time_minutes': float(point_remaining[row]),
                'true_activities': self._code(events[start + cut_point : start + real_lengths[row]]),
                'true_inter_event_time_minutes': self._waits(
                    true_waits[start + cut_point : start + real_lengths[row]]).tolist(),
                'true_remaining_time_minutes': float(true_elapsed[row] - prefix_elapsed[row]),
            })

        return rows

    def _decode(self, indices : np.ndarray) -> list[str]:
        """One run of activity indices, as the log's own names."""
        return [self.names[index] for index in indices.tolist()]

    def _code(self, indices : np.ndarray) -> str:
        """One activity run as suffix-generation's vocabulary-indexed character string."""
        return ''.join(self.codes[label] for label in self._decode(indices))

    def _draws(self,
               activities : np.ndarray,
               lengths : np.ndarray,
               minutes : np.ndarray,
               remaining : np.ndarray) -> dict:
        """Fold identical sampled activity suffixes while preserving the time of every draw."""
        suffixes = []
        positions = {}
        taken = []
        for sample in range(activities.shape[0]):
            suffix = self._code(activities[sample, :lengths[sample]])
            taken.append(positions.setdefault(suffix, len(positions)))
            if len(suffixes) == len(positions) - 1:
                suffixes.append(suffix)
        assert len(taken) <= np.iinfo(np.int16).max, 'Too many draws for generations schema'
        return {
            'generated_suffixes': suffixes,
            'generated_draws': taken,
            'generated_inter_event_time_minutes': [
                minutes[sample, :lengths[sample]].tolist() for sample in range(activities.shape[0])
            ],
            'generated_remaining_time_minutes': remaining.tolist(),
        }

    def _minutes(self, channel, encoded : np.ndarray) -> np.ndarray:
        """
        One channel's encoded values back in the log's own minutes, keeping the input's shape.

        ARGS:
        - channel: The `NumericChannel` the values were standardized against.
        - encoded: Standardized values, of any shape.
        """
        return channel.denormalize(encoded)

    def _waits(self, encoded : np.ndarray) -> np.ndarray:
        """
        Encoded `inter_event_time` back in minutes, clamped at zero.

        A wait runs forwards, so no time at all is the least one can be. The sampler already draws
        this channel above a raw zero, but the ground truth and the point prediction pass through
        here too.
        """
        return np.maximum(self._minutes(self.wait, encoded), 0.0)

    def _remaining(self, elapsed : np.ndarray, prefix_elapsed : np.ndarray) -> np.ndarray:
        """
        How long a suffix says the case still has to run.

        Clamped at zero: a suffix that ends on the event the prefix ended on has nothing left to run,
        and the decoder is under no constraint to keep the two apart.
        """
        return np.maximum(elapsed - prefix_elapsed, 0.0)
