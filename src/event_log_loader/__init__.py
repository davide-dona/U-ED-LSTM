"""
The read side of the format adapter: preprocessed splits as datasets the trainer consumes unchanged.

One stage per module, in the order a dataset is built:

- `spec.py`: which features exist and how the windows are shaped, off the preprocessing codec.
- `reader.py`: one split's own values, with the end-of-sequence events appended to every case.
- `encoding.py`: the vocabularies and statistics the features are encoded through, off the codec.
- `windows.py`: the prefix windows, and the `Dataset` serving them.
- `loader.py`: the four of them wired together.
"""

from .encoding import FeatureCodec
from .loader import PreSplitEventLogLoader
from .reader import add_eos_events, read_split
from .spec import DatasetSpec, read_codec, spec_from_codec
from .windows import WindowDataset, cut_points

__all__ = [
    'DatasetSpec',
    'FeatureCodec',
    'PreSplitEventLogLoader',
    'WindowDataset',
    'add_eos_events',
    'cut_points',
    'read_codec',
    'read_split',
    'spec_from_codec',
]
