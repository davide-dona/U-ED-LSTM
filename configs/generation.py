"""
How a generations run draws its suffixes.

These are the settings that decide what the sampled distribution *is*, as opposed to how fast it is
produced. They were defaults buried in `SuffixSampler.__init__` and could not be varied without
editing it; they live here so that a run can be varied from the command line, and so that
`scripts/generate.py` can stamp the values it used into the file it wrote. A generations file that
does not say how it was produced is not reproducible from the artifact alone.

Two constants deliberately do *not* live here. The log variance bounds a draw is read within are
`MIN_LOG_VARIANCE` / `MAX_LOG_VARIANCE` in `inference/sampler.py`, because they mirror the clamp in
`Loss.loss_attenuation_mse`: sampling outside the range the objective was defined on would draw from
a spread the model was never scored on, so they follow the loss rather than being tuned against it.
"""

from configs.event_log import CASE_ELAPSED_COLUMN, EVENT_ELAPSED_COLUMN

# Suffixes drawn per test prefix. A hundred is what the comparison's distribution metrics are read
# off, and what the other two models draw: the CVAE's `inference.evaluation_samples` and SuTraN's
# own `NUM_SAMPLES`. Ten, the smallest number hit-rate-at-10 can be read from, is what the runs
# before 2026-08-30 drew.
NUM_SAMPLES = 100

# Prefixes per batch, and hence per row group. Every prefix is decoded `NUM_SAMPLES` times over
# *within one tensor batch* (`SuffixSampler.sample` repeats the prefix along the batch), so this
# puts about thirteen thousand rows through the decoder at once. It was 1024 while `NUM_SAMPLES`
# was 10, which is the same width; a hundred draws at that many prefixes would be a hundred
# thousand. Throughput only: it does not change what is drawn.
BATCH_PREFIXES = 128

# Seed making the sampled suffixes reproducible. None leaves the global RNG alone, which is how the
# comparison's runs were drawn. Whatever it is, it is stamped into the generations file, so a file
# always says whether it can be reproduced.
SEED = None

# What `SuffixSampler` is built with, keyed by its own argument names, so that one dict configures
# the sampler and is stamped into the generations file.
SAMPLING = {
    # Whether a categorical draw perturbs the logits by their predicted spread, and a numerical one
    # its mean by the same. Together with dropout, these are what make a draw stochastic; switching
    # both off leaves only the dropout.
    'use_variance_cat': True,
    'use_variance_num': True,
    # Whether a categorical draw takes the argmax of the perturbed logits rather than sampling the
    # softmax over them. The argmax makes a draw the most likely event *given* the perturbation,
    # which narrows the distribution over suffixes considerably.
    'sample_argmax': False,
    # Whether the decoder keeps one dropout mask for the whole suffix. False redraws it every step,
    # which is the naive dropout the model is evaluated with.
    'variational_dropout_sampling': False,
    # Draws taken per step before a numerical value that has to stay positive gives up and reads
    # zero.
    'positive_retries': 10,
}

# Numerical columns a suffix may never decrease along: the time since the case started only runs
# forwards, and the comparison reads the remaining runtime off it.
GROWING_COLUMNS = (CASE_ELAPSED_COLUMN,)

# Numerical columns that may never fall below zero: no event follows the one before it in negative
# time.
POSITIVE_COLUMNS = (EVENT_ELAPSED_COLUMN,)
