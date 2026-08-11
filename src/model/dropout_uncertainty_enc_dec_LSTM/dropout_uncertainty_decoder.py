"""
Decoder consisting of two an two-layerd LSTM with LSTM cells using dropout as a Bayesian approximation.
"""

from .dropout_uncertainty_LSTM_cell import DropoutUncertaintyLSTMCell

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List

class DropoutUncertaintyLSTMDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_sizes: dict,
                 embeddings,
                 data_indices_dec,
                 num_layers: int,
                 dropout: Optional[float]=None):
        """
        Decoder part of the Encoder-Decoder LSTM.
        
        ARGS:
        - input_size: Size of input event attributes 
        - hidden_size: Size of hidden layers
        - output_size: Dictionary with sizes for output event attributes
        - embeddings: Categorical event attributes embeddings
        - data_indices_dec: Indices of event attributes
        - num_layers: Number of hidden layers in the LSTM
        - dropout: Dropout probability
        """
        super(DropoutUncertaintyLSTMDecoder, self).__init__()
        
        self.embeddings = embeddings
        
        self.data_indices_dec = data_indices_dec

        # Create a first cell:
        self.first_layer = DropoutUncertaintyLSTMCell(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        
        # Create multiple LSTM cells based on num_layer
        self.hidden_layers = nn.ModuleList([DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout) for i in range(num_layers-1)])
        
        self.output_sizes = output_sizes
        
        # Output_sizes is a list containing two dicts: one for categorical features and one for numerical features
        cat_output_sizes, num_output_sizes = output_sizes
        
        # Create a ModuleDict to hold the layers
        self.output_layers = nn.ModuleDict()
        # Dynamically create mean and variance output linear layers for categorical features and numerical features
        for key, size in cat_output_sizes.items():
            self.output_layers[f"{key}_mean"] = nn.Linear(hidden_size, size)
            self.output_layers[f"{key}_var"] = nn.Linear(hidden_size, size)
        
        for key, size in num_output_sizes.items():
            self.output_layers[f"{key}_mean"] = nn.Linear(hidden_size, size)
            self.output_layers[f"{key}_var"] = nn.Linear(hidden_size, size)
            
    def regularizer(self):
        """
        L2 regularization of Encoder weights, biases and dropout.
        
        OUTPUTS:
        - total_weight_reg: L2 weight regularization term
        - total_bias_reg: L2 bias regularization term
        - total_dropout_reg: L2 dropout regularization term
        """
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()
        
        for l in self.hidden_layers:
            weight, bias = l.regularizer()
            
            total_weight_reg += weight
            total_bias_reg += bias
            
        return total_weight_reg, total_bias_reg
    
    def __data_enc_for_model(self, data, pred):
        """
        Transform the dataloader input (prefix or suffix input) into a tensor structure for the encoder.
        
        INPUSTS:
        - data: previsous event data (either the last target or predicted).
        - pred: Boolean: true if predicted.
        
        OUTPUTS:
        - prefixes: Returns model input: Tensor seq_len x batch_size x input features (also embedded)
        """
        if pred:
            cats, nums = data
        else:
            cats = [data[0][i] for i in self.data_indices_dec[0]] # dims: list (n categorical values): Each with Tensor: batch_size x (window_size - suffix size)
            nums = [data[1][i] for i in self.data_indices_dec[1]] # dims: list (n numerical values): Each with Tensor: batch_size x (window_size - suffix size)
        
            assert len(cats) == len(self.data_indices_dec[0]) and len(nums) == len(self.data_indices_dec[1]), \
                f"Decoder: Number of input tensor is unequal the number of indices"
        
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
        
        # Merged input
        next_event = torch.cat((merged_cats, merged_nums), dim=-1).permute(1,0,2) # dim: seq_len x batch_size x input_features
        return next_event

    def forward(self, 
                input: Tensor,
                hx: Tuple[Tensor, Tensor],
                z: Optional[Tuple[List, List]] = None,
                pred: Optional[bool] = True) -> Tuple[list, Tuple[Tensor, Tensor]]:
        """
        Prediction of next event based on last hidden state and last event.
        
        INPUTS:
        - input_event: Either last sequence event or next predicted, target event: Tensor: seq_len (1) x batch_size x input_features
        - hx: Tuple containing last hidden state and cell state of the encoder: Tensor: batch_size x hidden_size
        
        OUTPUTS:
        - predictions: List containing activity mean, activity log variance, timestamp mean, timestamp log variance.
        - h, c: Updated hidden and cell states.
        """
        prediction_means = [{}, {}]  # List of two dicts: [cat_pred_means, num_pred_means]
        prediction_vars = [{}, {}]   # List of two dicts: [cat_pred_vars, num_pred_vars]

        # Process the input event through the encoder
        event = self.__data_enc_for_model(data=input, pred=pred)  # dim: Tensor: seq_len x batch_size x input feature

        # first decoder call initialize sample mask
        if z is None:
            z_hidden_layers = []
            # Pass input_event through the first LSTM layer and all hidden layers
            outputs, (h, c), z_first_layer = self.first_layer(input=event, hx=hx, z=None)
            
            for i, lstm_cell in enumerate(self.hidden_layers):
                outputs, (h, c), z_hidden_layer = lstm_cell(input=outputs, hx=(h, c), z=None)  
                z_hidden_layers.append(z_hidden_layer)
            
            z = (z_first_layer, z_hidden_layers) 
        # Use same sample masks from previous iterations for decoder
        else:
            # Pass input_event through the first LSTM layer and all hidden layers
            outputs, (h, c), _ = self.first_layer(input=event, hx=hx, z=z[0])
            
            for i, lstm_cell in enumerate(self.hidden_layers):
                outputs, (h, c), _ = lstm_cell(input=outputs, hx=(h, c), z=z[1][i])
            
        # Get the last output (outputs[-1]) for predictions
        final_output = outputs[-1]

        # Unpack output_sizes into categorical and numerical dicts
        cat_output_sizes, num_output_sizes = self.output_sizes

        # Predict means and variances for categorical features
        for key in cat_output_sizes:
            pred_mean = self.output_layers[f"{key}_mean"](final_output)
            prediction_means[0][f"{key}_mean"] = pred_mean  # Store in the first dict (for cat features)
            
            pred_var = self.output_layers[f"{key}_var"](final_output)
            prediction_vars[0][f"{key}_var"] = pred_var  # Store in the first dict (for cat features)

        # Predict means and variances for numerical features
        for key in num_output_sizes:
            pred_mean = self.output_layers[f"{key}_mean"](final_output)
            prediction_means[1][f"{key}_mean"] = pred_mean
            
            pred_var = self.output_layers[f"{key}_var"](final_output)
            prediction_vars[1][f"{key}_var"] = pred_var

            # Get raw log-variance prediction
            # raw_logvar = self.output_layers[f"{key}_var"](final_output)
            # LOGVAR_BOUND: 3.0 -> var = [exp(-3.0)=0.05, exp(3.0)=20]
            # LOGVAR_BOUND = 6.0
            # Bound the log-variance using tanh activation
            # bounded_logvar = LOGVAR_BOUND * torch.tanh(raw_logvar / LOGVAR_BOUND)
            # prediction_vars[1][f"{key}_var"] = bounded_logvar  # Store the safe version
            
        predictions = [prediction_means, prediction_vars]

        # Return the prediction dictionaries for means and variances along with the hidden states
        return predictions, (h, c), z