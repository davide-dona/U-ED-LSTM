"""
LSTM cells using dropout as a Bayesian approximation.

The four gates are held as one stacked parameter each for the input and the hidden projection,
rather than as eight separate `nn.Linear` modules. The arithmetic is unchanged: gate `g` still reads
`W_g (x * zx_g) + b_g + U_g (h * zh_g) + c_g`, with its own dropout mask and its own two bias
vectors. Stacking only changes how many kernels that costs. See `forward` for the layout.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

# Gate order along the stacked parameters' leading dimension. Input, forget and output take a
# sigmoid and the cell candidate a tanh, so keeping the candidate last lets the three sigmoids share
# a single contiguous slice.
GATES = ('i', 'f', 'o', 'c')
NUM_GATES = len(GATES)


class DropoutUncertaintyLSTMCell(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: Optional[float]=None):
        """
        ARGS:
        - input_size: Size of input features
        - hidden_size: Size of hidden layer
        - dropout: should be between 0 and 1
        """
        super(DropoutUncertaintyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize dropout
        if dropout is None:
            # Set p for dropout to random parameter
            self.p_logit = nn.Parameter(torch.empty(1).normal_())
        elif not 0 <= dropout < 1:
            # p dropout must be between 0 and 1
            raise Exception("Dropout rate should be between in [0, 1)")
        else:
            # Set p dropout to the fixed value
            self.p_logit = dropout

        # One stacked parameter per projection, gate-major. The trailing two dimensions are
        # (in, out) rather than `nn.Linear`'s (out, in), so that a gate is a plain `bmm` slice.
        self.weight_ih = nn.Parameter(torch.empty(NUM_GATES, input_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(NUM_GATES, 1, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(NUM_GATES, hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(NUM_GATES, 1, hidden_size))

        self.init_weights()

    def init_weights(self):
        """
        Initializes weight layers with initial values
        """
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()

        for parameter in (self.weight_ih, self.bias_ih, self.weight_hh, self.bias_hh):
            parameter.data.uniform_(-k, k)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """
        Accept checkpoints written before the gates were stacked.

        Those hold eight `nn.Linear` submodules, `W{i,f,c,o}` and `U{i,f,c,o}`. They are folded into
        the stacked parameters here, transposing each weight into (in, out), so that the pretrained
        checkpoints keep loading against the current module.
        """
        if f'{prefix}Wi.weight' not in state_dict:
            return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

        for stacked, legacy in (('weight_ih', 'W'), ('weight_hh', 'U')):
            state_dict[prefix + stacked] = torch.stack(
                [state_dict.pop(f'{prefix}{legacy}{gate}.weight').t() for gate in GATES])
        for stacked, legacy in (('bias_ih', 'W'), ('bias_hh', 'U')):
            state_dict[prefix + stacked] = torch.stack(
                [state_dict.pop(f'{prefix}{legacy}{gate}.bias').unsqueeze(0) for gate in GATES])

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _dropout_probability(self):
        """
        The dropout probability, as a float when it is fixed and as a tensor when it is learned.
        """
        if isinstance(self.p_logit, float):
            return self.p_logit
        return torch.sigmoid(self.p_logit)

    def _sample_mask(self, B, device):
        """
        Applies dropout to the LSTM Cell weight layers

        INPUTS:
        B: Batch size
        device: Device the masks are drawn on, i.e. the one the input lives on.

        OUTPUTS:
        zx: Dropout mask for weight layer before input, dim: gates x batch_size x input_size
        zh: Dropout mask for weight layer before hidden, dim: gates x batch_size x hidden_size

        Note: value p_logit at infinity can cause numerical instability. Dropout masks for 4 gates, scale input by 1 / (1 - p)
        """
        # Check dropout probability
        p = self._dropout_probability()

        eps = 1e-7
        t = 1e-1

        # tensors with random values:
        ux = torch.rand(NUM_GATES, B, self.input_size, device=device) # dim gates x batch_size x input_size
        uh = torch.rand(NUM_GATES, B, self.hidden_size, device=device)  # dim (gates=weight matrices per cell x batch_size x hidden_size)

        # The p-dependent half of the concrete-dropout logit is the same for every element, so it is
        # folded into one scalar rather than broadcast through two logs per mask.
        if isinstance(p, float):
            logit_p = math.log(p + eps) - math.log(1 - p + eps)
        else:
            logit_p = torch.log(p + eps) - torch.log(1 - p + eps)

        # Dropout masks: containing values near 1 for keeping weights, and near 0 for dropping weights for each gate and batch
        if self.input_size == 1:
            logit_eps = math.log(eps) - math.log(1 + eps)
            zx = (1-torch.sigmoid((logit_eps + torch.log(ux+eps) - torch.log(1-ux+eps))/ t))
        else:
            # dim: gates x batch_size x input_features
            zx = (1-torch.sigmoid((logit_p + torch.log(ux+eps) - torch.log(1-ux+eps))/ t)) / (1-p)
        # dim: gates x batch_size x input_features
        zh = (1-torch.sigmoid((logit_p + torch.log(uh+eps) - torch.log(1-uh+eps))/ t)) / (1-p)

        return zx, zh

    def regularizer(self):
        """
        L2 regularization of weights and biases scaled for dropout
        """
        # Compute dropout probability
        p = self._dropout_probability()

        # Weight L2 sum (keeps autograd)
        weight_sum = (torch.sum(self.weight_ih ** 2) + torch.sum(self.weight_hh ** 2)) / (1. - p)

        # Bias L2 sum
        bias_sum = torch.sum(self.bias_ih ** 2) + torch.sum(self.bias_hh ** 2)

        return weight_sum, bias_sum

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None,
                z: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        INPUTS:
        - input: Input tensor with shape (sequence, batch, input dimension)
        - hx: h_t: hidden state and c_t: cell state as tuple at time step (event t)
        - z: dropout masks for LSTM weights

        OUTPUTS:
        - hn: List of all hidden states: h_1, ... h_n
        - (h_t, c_t): Last hidden and cell state
        """
        # Determine device
        device = input.device

        # seq_len
        T = input.shape[0]
        # batch_size
        B = input.shape[1]

        # Initialize hidden and cell states
        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, dtype=input.dtype, device=device)  # Ensure device is correct
            c_t = torch.zeros(B, self.hidden_size, dtype=input.dtype, device=device)
        else:
            # follow up layer
            h_t = hx[0]
            c_t = hx[1]

        if z is None:
            # Masks
            zx, zh = self._sample_mask(B, device)
        else:
            zx, zh = z

        # The input projection does not depend on the recurrence, and the dropout mask is shared
        # across time steps, so every gate at every time step is computed in one `baddbmm` here
        # instead of one `Linear` per gate per step inside the loop. Both bias vectors of a gate are
        # folded in at the same time, which leaves the loop with a single fused matmul per step.
        # dim: gates x seq_len x batch_size x input_size
        masked_input = input.unsqueeze(0) * zx.unsqueeze(1)
        gates_x = torch.baddbmm(self.bias_ih + self.bias_hh,
                                masked_input.reshape(NUM_GATES, T * B, self.input_size),
                                self.weight_ih).view(NUM_GATES, T, B, self.hidden_size)
        # Split the time steps up front. Indexing `gates_x` inside the loop instead would make each
        # step's backward allocate a zero tensor the size of the whole projection to scatter into.
        gates_x = gates_x.unbind(1)

        # Store all the hidden states for each time step (t=1, ..., T) for all events in prefix for each batch
        hn = []

        # Time-step loop: Iterate over each event in the prefix:
        for t in range(T):
            # Drop out random hidden values, one mask per gate
            h_gates = h_t.unsqueeze(0) * zh

            # Compute LSTM gates: input projection, both biases and hidden projection in one kernel
            gates = torch.baddbmm(gates_x[t], h_gates, self.weight_hh)

            # Input gate: store new information. Forget gate: which information from the previous
            # step is kept and which thrown away. Output gate: what of the cell state is exposed.
            g_i, g_f, g_o, g_c = gates.unbind(0)
            i, f, o = torch.sigmoid(g_i), torch.sigmoid(g_f), torch.sigmoid(g_o)
            c_tilde = torch.tanh(g_c)

            # Updated cell state
            c_t = torch.addcmul(f * c_t, i, c_tilde)
            # Updated hidden state
            h_t = o * torch.tanh(c_t)
            # Output = output * tanh(cell state): hidden output state for all n events for each batch
            hn.append(h_t)

        # dim: seq_len x batch_size x hidden size
        hn = torch.stack(hn)

        return hn, (h_t, c_t), (zx, zh)
