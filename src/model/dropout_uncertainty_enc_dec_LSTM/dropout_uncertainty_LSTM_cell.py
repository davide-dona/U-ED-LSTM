"""
LSTM cells using dropout as a Bayesian approximation.
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple

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

        # Input gate
        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        # Forget gate
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        # Cell state gate
        self.Wc = nn.Linear(self.input_size, self.hidden_size)
        self.Uc = nn.Linear(self.hidden_size, self.hidden_size)
        # Output gate
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
                
        self.init_weights()
    
    def init_weights(self):
        """
        Initializes weight layers with initial values
        """
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()
        
        # Input gate weights:
        self.Wi.weight.data.uniform_(-k,k)
        self.Wi.bias.data.uniform_(-k,k)
        self.Ui.weight.data.uniform_(-k,k)
        self.Ui.bias.data.uniform_(-k,k)
        
        # Forget gate weights
        self.Wf.weight.data.uniform_(-k,k)
        self.Wf.bias.data.uniform_(-k,k)
        self.Uf.weight.data.uniform_(-k,k)
        self.Uf.bias.data.uniform_(-k,k)
        
        # Cell state gate weights
        self.Wc.weight.data.uniform_(-k,k)
        self.Wc.bias.data.uniform_(-k,k)
        self.Uc.weight.data.uniform_(-k,k)
        self.Uc.bias.data.uniform_(-k,k)
        
        # Output gate weights
        self.Wo.weight.data.uniform_(-k,k)
        self.Wo.bias.data.uniform_(-k,k)
        self.Uo.weight.data.uniform_(-k,k)
        self.Uo.bias.data.uniform_(-k,k)
        
    def _sample_mask(self, B):
        """
        Applies dropout to the LSTM Cell weight layers
        
        INPUTS:
        B: Batch size

        OUTPUTS:
        zx: Dropout mask for weight layer before input
        zh: Dropout mask for weight layer before hidden
        
        Note: value p_logit at infinity can cause numerical instability. Dropout masks for 4 gates, scale input by 1 / (1 - p)
        """
        # Check dropout probability
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)

        # Four Weight matrix pairs: Perform dropout for each weight layer.
        GATES = 4
        
        eps = torch.tensor(1e-7)
        t = 1e-1

        # tensors with random values: 
        ux = torch.rand(GATES, B, self.input_size) # dim gates x batch_size x input_size
        uh = torch.rand(GATES, B, self.hidden_size)  # dim (gates=weight matrices per cell x batch_size x hidden_size)

        # Dropout masks: containing values near 1 for keeping weights, and near 0 for dropping weights for each gate and batch
        if self.input_size == 1:
            zx = (1-torch.sigmoid((torch.log(eps) - torch.log(1+eps)+ torch.log(ux+eps) - torch.log(1-ux+eps))/ t))
        else:
            # dim: gates x batch_size x input_features
            zx = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps) + torch.log(ux+eps) - torch.log(1-ux+eps))/ t)) / (1-p)
        # dim: gates x batch_size x input_features
        zh = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)+ torch.log(uh+eps) - torch.log(1-uh+eps))/ t)) / (1-p)

        return zx, zh

    def regularizer(self):
        """
        L2 regularization of weights and biases scaled for dropout
        """
        # Compute dropout probability
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)

        # Weight L2 sum (keeps autograd)
        weight_sum = sum(torch.sum(params ** 2) 
                        for name, params in self.named_parameters() 
                        if name.endswith("weight")) / (1. - p)

        # Bias L2 sum
        bias_sum = sum(torch.sum(params ** 2) 
                    for name, params in self.named_parameters() 
                    if name.endswith("bias"))

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

        # Store all the hidden states for each time step (t=1, ..., T) for all events in prefix for each batch
        hn = torch.empty(T, B, self.hidden_size, dtype=input.dtype, device=device) # dim: seq_len x batch_size x hidden size
        
        if z is None:
            # Masks
            zx, zh = self._sample_mask(B)
            zx = [mask.to(device) for mask in zx]  
            zh = [mask.to(device) for mask in zh]
            (zx, zh)
        else:
            zx, zh = z
    
        # Time-step loop: Iterate over each event in the prefix:
        for t in range(T):
            # Drop out randm input and hidden values
            x_i, x_f, x_c, x_o = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_c, h_o = (h_t * zh_ for zh_ in zh)
    
            # Compute LSTM gates
            # Input gate: Store new information
            i = torch.sigmoid(self.Wi(x_i) + self.Ui(h_i))
            # Forget gate: Which information from previous step is kept and which thrown away
            f = torch.sigmoid(self.Wf(x_f) + self.Uf(h_f))
            c_tilde = torch.tanh(self.Wc(x_c) + self.Uc(h_c))
            o = torch.sigmoid(self.Wo(x_o) + self.Uo(h_o))
    
            # Updated cell state
            c_t = f * c_t + i * c_tilde
            # Updated hidden state
            h_t = o * torch.tanh(c_t)
            # Output = output * tanh(cell state): hidden output state for all n events for each batch
            hn[t] = h_t
    
        return hn, (h_t, c_t), (zx, zh)