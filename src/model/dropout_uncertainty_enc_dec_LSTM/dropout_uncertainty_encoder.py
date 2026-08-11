"""
Encoder consisting of two an two-layerd LSTM with LSTM cells using dropout as a Bayesian approximation.
"""

from .dropout_uncertainty_LSTM_cell import DropoutUncertaintyLSTMCell

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List

class DropoutUncertaintyLSTMEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 embeddings,
                 data_indices_enc: list,
                 num_layers: int,
                 dropout: Optional[float]=None):
        """
        Encoder part of the Encoder-Decoder LSTM
        
        ARGS:
        - input_size: Size of input features
        - hidden_size: Size of hidden layer
        - embeddings: Embedding modules for categorical data encoder
        - data_indices_enc: Indicies of tensors used as input of the encoder model
        - num_layers: Number hidden layers in the LSTM
        - dropout: Dropout probability
        """
        super(DropoutUncertaintyLSTMEncoder, self).__init__()
        
        # Encoder learnable embeddings
        self.embeddings = embeddings
        
        # List of two lists (categorical, numerical) each containing the indices of tensors required for encoder
        self.data_indices_enc = data_indices_enc
        
        # Create a first cell:
        self.first_layer = DropoutUncertaintyLSTMCell(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        # Create multiple LSTM cells based on num_layer
        self.hidden_layers = nn.ModuleList([DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout) for i in range(num_layers-1)])

    def regularizer(self) -> Tuple[float, float]:
        """
        L2 regularization of Encoder weights, biases and dropout.
        
        OUTPUTS:
        - total_weight_reg: L2 weight regularization term
        - total_bias_reg: L2 bias regularization term
        """
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()
        
        for l in self.hidden_layers:
            weight, bias = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias
            
        return total_weight_reg, total_bias_reg
        
    def forward(self, input: List) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the encoder to get the final (hidden) state vector as input for the decoder.
        
        INPUTS:
        - input: Prefixes, Tensor: seq_len, batch_size, input_size
        
        OUTPUTS:
        - h,c: Last hidden and cell states of the last layer.
        """
        
        # Transform the input into 
        prefixes = self.__data_enc_for_model(data=input) # dim: Tensor: seq_len x batch_size x input feature (cat as embedding) 
        
        # Outputs: All hidden states of all cells in the layer, h,c: last hidden state and cell state in the layer
        outputs, (h, c), _ = self.first_layer(input=prefixes, hx=None, z=None)
        
        # Pass through the remaining LSTM cell: Layer gets for: input: h_n Tensor, hx: (h, c)
        for _, layer in enumerate(self.hidden_layers):
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None)
  
        return (h, c)
    
    def __data_enc_for_model(self, data):
        """
        Transform the dataloader input (prefix or suffix input) into a tensor structure for the encoder.
        
        INPUTS:
        - data: dataloader input
        
        OUTPUTs:
        - prefixes: Returns model input: Tensor seq_len x batch_size x input features (also embedded)
        """       
        cats = [data[0][i] for i in self.data_indices_enc[0]] # dims: list (n categorical values): Each with Tensor: batch_size x (window_size - suffix size)
        nums = [data[1][i] for i in self.data_indices_enc[1]] # dims: list (n numerical values): Each with Tensor: batch_size x (window_size - suffix size)
        
        assert len(cats) == len(self.data_indices_enc[0]) and len(nums) == len(self.data_indices_enc[1]), \
            f"Encoder: Number of input tensor is unequal the number of indices"
                
        # Embedd categorical tensors
        embedded_cats = []
        for i, embedd in enumerate(self.embeddings):        
            embedded_cats.append(embedd(cats[i]))

        # Merged categroical data
        merged_cats = torch.cat([cat for cat in embedded_cats], dim=-1)
    
        if len(nums):
            # Merged numerical inputs
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=merged_cats.device)
        prefixes = torch.cat((merged_cats, merged_nums), dim=-1).permute(1,0,2) # dim: seq_len x batch_size x input_features
        return prefixes