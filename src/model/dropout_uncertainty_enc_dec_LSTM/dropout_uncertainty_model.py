"""
Enocder-Decoder LSTM modelling epistemic uncertainty via dropout as a Bayesian approximation.
- During training encoder and decoder use Variational Dropout.
- During testing encoder uses Variational Dropout, Decoder uses Naive Dropout.
"""

from .dropout_uncertainty_encoder import DropoutUncertaintyLSTMEncoder
from .dropout_uncertainty_decoder import DropoutUncertaintyLSTMDecoder

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List

class DropoutUncertaintyEncoderDecoderLSTM(nn.Module):
    def __init__(self,
                 data_set_categories : list[tuple[str, dict[str, int]]],
                 enc_feat: list,
                 dec_feat: list,
                 seq_len_pred: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: Optional[float]=None):
        
        """
        Full Encoder-Decoder architecture with droput uncertainty LSTM
        
        ARGS:
        - data_set_categories: Event attributes, name and size
        - enc_feat: Event attributes used be encoder as input
        - dec_feat Event attributes used be decoder as input and output
        - num_layers: Number of hidden layers in both Encoder and Decoder
        - dropout: Dropout probability
        """
        
        super(DropoutUncertaintyEncoderDecoderLSTM, self).__init__()
        
        # Feature sizes
        self.data_set_categories = data_set_categories
        print("Data set categories: ", data_set_categories)
        self.enc_feat = enc_feat
        print("Encoder input features: ", enc_feat)
        self.dec_feat = dec_feat
        print("Decoder input+output features: ", dec_feat)
        
        print("\n")
        
        # Sequence lenght prediciton
        self.seq_len_pred = seq_len_pred
        print("Sequence length of decoder output: ", seq_len_pred)
        
        print("\n")
        
        # Parameters for encoder and decoder
        self.hidden_size = hidden_size
        print("Cells hidden size: ", hidden_size)
        self.num_layers = num_layers
        print("Number of LSTM layer: ", num_layers)
        self.dropout = dropout
        print("Dropout rate: ", dropout)
        
        print("\n")
        
        # Encoder
        # Get list of category label values for cat and num
        enc_label_cats, enc_label_nums = self.__get_list_labels_input(data_set_categories=data_set_categories, model_type_feats=enc_feat)
        self.data_labels_features_enc = [enc_label_cats, enc_label_nums]
        print("Encoder number of labels for each input feature (categorical, numerical): ", self.data_labels_features_enc)
        
        data_cat_indices_enc, data_num_indices_enc = self.__get_list_tensor_indeces(data_set_categories=data_set_categories, model_type_feats=enc_feat)
        self.data_indices_enc = [data_cat_indices_enc, data_num_indices_enc]
        print("Encoder indices of tensors in dataset used as input: ", self.data_indices_enc)
        
        # Create embeddings for categorical features
        self.embeddings_enc = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in enc_label_cats])
        print("Embeddings encoder: ", self.embeddings_enc)
        
        # Compute total input size encoder
        embedding_size_enc = sum([min(600, round(1.6 * n_cat**0.56)) for n_cat in enc_label_cats])
        print("Total embedding feature size encoder: ", embedding_size_enc)
        
        num_size_enc = sum(enc_label_nums)
        print("Total numerical feature size encoder: ", num_size_enc)
        
        self.input_size_enc = embedding_size_enc + num_size_enc 
        print("Input feature size encoder: ", self.input_size_enc)       
        
        # Define Encoder
        self.encoder = DropoutUncertaintyLSTMEncoder(input_size=self.input_size_enc,
                                                     hidden_size=hidden_size,
                                                     embeddings=self.embeddings_enc,
                                                     data_indices_enc=self.data_indices_enc, 
                                                     num_layers=num_layers,
                                                     dropout=dropout)
        print("Encoder initialized! \n")
        
        # Decoder
        # Get list of category label values for cat and num
        dec_label_cats, dec_label_nums = self.__get_list_labels_input(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        print("Decoder label values size for each categorical input feature: ", dec_label_cats)
        print("Decoder label values size for each numerical input feature: ", dec_label_nums)
            
        data_cat_indices_dec, data_num_indices_dec = self.__get_list_tensor_indeces(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        self.data_indices_dec = [data_cat_indices_dec, data_num_indices_dec]
        print("Decoder indices of tensors in dataset used as input: ", self.data_indices_dec)
        
        # Create embeddings for categorical features
        self.embeddings_dec = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in dec_label_cats])
        print("Embeddings decoder: ", self.embeddings_dec)
        
        # Compute total input size decoder
        embedding_size_dec = sum([min(600, round(1.6 * n_cat**0.56)) for n_cat in dec_label_cats])
        print("Total embedding feature size decoder: ", embedding_size_dec)
        
        num_size_dec = sum(dec_label_nums)
        print("Total numerical feature size decoder: ", num_size_dec)
        
        self.input_size_dec = embedding_size_dec + num_size_dec
        print("Input feature size decoder: ", self.input_size_dec)
        
        # Dictionary of output features and output_sizes
        self.output_sizes = self.__get_list_dict_labels_output(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        print("Output feature list of dicts (featue name, feature output size) of decoder: ", self.output_sizes)
        
        # Define Decoder
        self.decoder = DropoutUncertaintyLSTMDecoder(input_size=self.input_size_dec,
                                                     hidden_size=hidden_size,
                                                     output_sizes=self.output_sizes,
                                                     embeddings=self.embeddings_dec,
                                                     data_indices_dec=self.data_indices_dec,
                                                     num_layers=num_layers,
                                                     dropout=dropout)
        print("Decoder initialized! \n")
        
        # List containing two dicts: One for categorical, one for numerical
        self.output_feature_indeces = self.__get_list_dict_feature_index(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        # print("Output feature list of dicts (featue name, tensor index in dataset) of decoder: ", self.output_feature_indeces)
        
    def __get_list_labels_input(self,
                                data_set_categories,
                                model_type_feats):
        """
        Returns two lists (categorical, numerical) containing the number of feature attributes (e.g.: 28 activity labels for concept:name)
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}

        # Get input feature sizes to determine the model input size
        label_cats = [cat_dict[cat_feat] for cat_feat in cat_feat_model if cat_feat in cat_dict]
        label_nums = [num_dict[num_feat] for num_feat in num_feat_model if num_feat in num_dict]

        return label_cats, label_nums
    
    def __get_list_tensor_indeces(self,
                                  data_set_categories,
                                  model_type_feats):
        """
        Returns two lists (categorical, numerical) containing the indices of the tensors in the datset used as input for the encoder.
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Convert cat_feat_model and num_feat_model to sets for O(1) membership checks
        cat_feat_set = set(cat_feat_model)
        num_feat_set = set(num_feat_model)

        # Get indices of tensors used as input of model
        cat_indices = [i for i, cat in enumerate(cat_categories) if cat[0] in cat_feat_set]
        num_indices = [i for i, num in enumerate(num_categories) if num[0] in num_feat_set]

        return cat_indices, num_indices
    
    def __get_list_dict_labels_output(self,
                                      data_set_categories,
                                      model_type_feats):
        """
        Returns a list of two dicts (categorical, numerical) containing the key: feature name and the value: number of labels of feature.
        Decoder, Output only!
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}

        # Create separate dictionaries for categorical and numerical features
        cat_labels_dict = {cat_feat: cat_dict[cat_feat] for cat_feat in cat_feat_model if cat_feat in cat_dict}
        num_labels_dict = {num_feat: num_dict[num_feat] for num_feat in num_feat_model if num_feat in num_dict}

        # Return a list containing two dicts: one for categorical and one for numerical features
        return [cat_labels_dict, num_labels_dict]
    
    def __get_list_dict_feature_index(self, data_set_categories, model_type_feats):
        """
        Returns a list of two dicts (categorical, numerical) containing the key: feature name and the value: indices of the tensors in the datset used as input for the encoder.
        Decoder, Output only!
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Convert cat_feat_model and num_feat_model to sets for O(1) membership checks
        cat_feat_set = set(cat_feat_model)
        num_feat_set = set(num_feat_model)

        # Create dictionaries to store feature names and their index positions
        cat_index_dict = {cat[0]: i for i, cat in enumerate(cat_categories) if cat[0] in cat_feat_set}
        num_index_dict = {num[0]: i for i, num in enumerate(num_categories) if num[0] in num_feat_set}

        # Return a list of two dicts: one for categorical and one for numerical features
        return [cat_index_dict, num_index_dict]
               
    def forward(self, prefixes: List, suffixes: Optional[List]=None, teacher_forcing_ratio: Optional[float]=0.0):
        """
        Full forward pass through the Encoder-Decoder architecture
        
        INPUTS:
        - prefixes: Input prefix sequence: list(list(tensor(categorical), list(tensor(numerical)))
        - suffixes: Suffix to predict: Tensor: list(list(tensor(categorical), list(tensor(numerical)))
        - teacher_forcing_ratio: Value between 0 and 1 to select pred or target as last event.
        
        OUTPUTS:
        - predictions: Predicted outcome. [categorical dict (key: feature name, value tensor), numerical dict (key: feature name, value tensor)]
        - (h,c): Predicted last hidden and cell state.
        - self.seq_len_pred: Sequence length.
        - self.output_feature_indeces: Target data indices: [categorical dict(key: feature name, value: index of tensor in categorical list of dataset), 
                                                             numerical dict(key: feature name, value: index of tensor in numerical list of dataset)]
        
        """
        # Model is in training mode and suffixes are provided
        training = self.training and suffixes is not None
        # Model is in evaluation (validation) mode and suffixes are provided
        validation = not self.training and suffixes is not None              
        
        # Call encoder
        (h_enc, c_enc) = self.encoder(input=prefixes)
                
        # Get SOS event: Last prefx event:
        cat_prefixes, num_prefixes = prefixes
        cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
        num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
        sos_event = [cat_sos_events, num_sos_events]
                
        # output_sizes is a list of two dicts: [cat_dict, num_dict]
        cat_output_features_labels, num_output_features_labels = self.output_sizes
        # Prediction dictionary for categorical features
        cat_predictions = {f"{key}_{suffix}": None for key in cat_output_features_labels for suffix in ["mean", "var"]}
        # Prediction dictionary for numerical features
        num_predictions = {f"{key}_{suffix}": None for key in num_output_features_labels for suffix in ["mean", "var"]}
        predictions = [cat_predictions, num_predictions]
    
        # Training
        if training:
            # Timestep iterations
            for t in range(self.seq_len_pred):                
                # SOS Event
                if t == 0:
                    # preds: list containing two dicts one for all means (cat, num), one for all vars (cat, num)
                    preds, (h, c), z = self.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)
                    pred_means, pred_vars = preds
                # Next Event
                # Decide per timestep whether to use teacher forcing
                else:
                    # Random vale for teacher forcing for each timestep: If smaller use target else predicted
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Use ground-truth previous event
                        cat_t_suffix_event = [cat_tens[:, t-1:t] for cat_tens in suffixes[0]]
                        num_t_suffix_event = [num_tens[:, t-1:t] for num_tens in suffixes[1]]
                        t_suffix_event = [cat_t_suffix_event, num_t_suffix_event]
                        
                        preds, (h, c), _ = self.decoder(input=t_suffix_event, hx=(h, c), z=z, pred=False)
                        pred_means, pred_vars = preds
                    else:
                        # Use model prediction
                        last_pred_event = self.__transform_pred_into_next_event(pred_means=pred_means, pred_index=t, suffix=suffixes)
                        preds, (h, c), _ = self.decoder(input=last_pred_event, hx=(h, c), z=z, pred=True)
                        pred_means, pred_vars = preds

                cat_pred_means, num_pred_means = pred_means
                cat_pred_vars, num_pred_vars = pred_vars
                         
                # Add categorical tensors to output
                for key in cat_output_features_labels:  
                    if t == 0:
                        predictions[0][f"{key}_mean"] = cat_pred_means[f"{key}_mean"].unsqueeze(0)
                        predictions[0][f"{key}_var"] = cat_pred_vars[f"{key}_var"].unsqueeze(0)
                    else:
                        predictions[0][f"{key}_mean"] = torch.cat((predictions[0][f"{key}_mean"], cat_pred_means[f"{key}_mean"].unsqueeze(0)), dim=0)
                        predictions[0][f"{key}_var"] = torch.cat((predictions[0][f"{key}_var"], cat_pred_vars[f"{key}_var"].unsqueeze(0)), dim=0)

                # Add numerical tensors to output
                for key in num_output_features_labels:
                    if t == 0:
                        predictions[1][f"{key}_mean"] = num_pred_means[f"{key}_mean"].unsqueeze(0)
                        predictions[1][f"{key}_var"] = num_pred_vars[f"{key}_var"].unsqueeze(0)
                    else:
                        predictions[1][f"{key}_mean"] = torch.cat((predictions[1][f"{key}_mean"], num_pred_means[f"{key}_mean"].unsqueeze(0)), dim=0)
                        predictions[1][f"{key}_var"] = torch.cat((predictions[1][f"{key}_var"], num_pred_vars[f"{key}_var"].unsqueeze(0)), dim=0)
                        
        # Validation:
        if validation:
            for k in range(self.seq_len_pred):
                if k == 0:
                    preds, (h, c), z = self.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)
                    pred_means, pred_vars = preds
                else:
                    last_pred_event = self.__transform_pred_into_next_event(pred_means=pred_means)
                    preds, (h, c), z = self.decoder(input=last_pred_event, hx=(h, c), z=z, pred=True)                    
                    pred_means, pred_vars = preds
                    
                cat_pred_means, num_pred_means = pred_means
                cat_pred_vars, num_pred_vars = pred_vars
                        
                # Add categorical tensors to output
                for key in cat_output_features_labels:  
                    if k == 0:
                        predictions[0][f"{key}_mean"] = cat_pred_means[f"{key}_mean"].unsqueeze(0)
                        predictions[0][f"{key}_var"] = cat_pred_vars[f"{key}_var"].unsqueeze(0)
                    else:
                        predictions[0][f"{key}_mean"] = torch.cat((predictions[0][f"{key}_mean"], cat_pred_means[f"{key}_mean"].unsqueeze(0)), dim=0)
                        predictions[0][f"{key}_var"] = torch.cat((predictions[0][f"{key}_var"], cat_pred_vars[f"{key}_var"].unsqueeze(0)), dim=0)

                # Add numerical tensors to output
                for key in num_output_features_labels:
                    if k == 0:
                        predictions[1][f"{key}_mean"] = num_pred_means[f"{key}_mean"].unsqueeze(0)
                        predictions[1][f"{key}_var"] = num_pred_vars[f"{key}_var"].unsqueeze(0)
                    else:
                        predictions[1][f"{key}_mean"] = torch.cat((predictions[1][f"{key}_mean"], num_pred_means[f"{key}_mean"].unsqueeze(0)), dim=0)
                        predictions[1][f"{key}_var"] = torch.cat((predictions[1][f"{key}_var"], num_pred_vars[f"{key}_var"].unsqueeze(0)), dim=0)
    
        # Return training or validation output
        return predictions, (h,c), self.seq_len_pred, self.output_feature_indeces
    
    def __transform_pred_into_next_event(self, pred_means, pred_index:Optional[int]=None, suffix:Optional[list]=None):
        """
        Gets the predicted values (means) and transform it into input for decoder model
        -> input: list(list(categorical tensors(batch size x 1)), list(numerical tensors(batch size x 1)))
        
        INPUTS: 
        - pred_means: predicted values
        - pred_index: index of event for next prediction
        - suffix: Target data
        
        OUTPUTS:
        - next_event: event in decoder input data format
        """
        cat_pred_means, num_pred_means = pred_means 
        
        # Create index tensor based on predicted logits
        cat_preds = [torch.argmax(tensor, dim=1).unsqueeze(1) for _, tensor in enumerate(cat_pred_means.values())]
        
        # Create time tesnors (if training: next preiction time must always be >= last prediction)
        if pred_index == None or pred_index == 0:
            num_preds = [pred_means for _, pred_means in enumerate(num_pred_means.values())]
        # Replace num preds with previous target event in case the predicted one is smaller the previous target events one
        else:
            assert suffix is not None, \
                f"Suffix is None"
            # Get numerical targets
            _, num_suffix = suffix
            # Get the previous true event tesnors (which are the inputs for the decoder)
            num_suffix_dec = [num_suffix[i][:, pred_index-1:pred_index] for i in self.data_indices_dec[1]]        
            
            # Get the predicted event tensors
            num_preds_list = [pred_means for _, pred_means in enumerate(num_pred_means.values())]
            
            assert len(num_preds_list) == len(num_suffix_dec), \
                f"Not same lenght of elements in previous decoder and prediction numeric values."
                
            num_preds = [torch.max(num_suffix_dec[i], num_preds_list[i]) for i in range(len(num_suffix_dec))]

        last_event = [cat_preds, num_preds]
                
        return last_event
    
    def inference(self,
                  prefix: Optional[list]=None,
                  last_event: Optional[list]=None,
                  hx: Optional[Tuple[Tensor, Tensor]]=None,
                  z: Optional[Tuple[List, List]]=None):
        """
        Inference method fo scenario analysis based on Monte Carlo sampling.
        
        INPUTS:
        - prefix: Input sequence of the model to be analyzed by encoder. (Set param only for the first model call)
        - last_event: Last event which was the output of the decoder. (Set param only after the first model call)
        - hx: Last hidden state which was the output of the decoder. (Set param only for the first model call) 
        
        OUTPUTS:
        - predictions: Predicted outcome. [categorical dict (key: feature name, value tensor), numerical dict (key: feature name, value tensor)]
        - (h,c): Predicted last hidden and cell state
        """
        with torch.no_grad():
            # First Prediciton
            if prefix is not None:
                # Call encoder
                (h_enc, c_enc) = self.encoder(input=prefix)
                        
                # Get SOS event: Last prefx event:
                cat_prefixes, num_prefixes = prefix
                cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
                num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
                sos_event = [cat_sos_events, num_sos_events]
                
                preds, (h, c), z = self.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)
                
                # Return the sample masks for consistent variational inference
                return preds, (h, c), z
              
            # Second-n_th prediction
            else:
                (h, c) = hx
                preds, (h, c), _ = self.decoder(input=last_event, hx=(h, c), z=z, pred=True)
                return preds, (h, c)
                
    def save(self, path : str):
        """
        Store the trained model at path.
        """
        checkpoint = {'model_state_dict' : self.state_dict(),
                      'kwargs' : {
                        'data_set_categories' : self.data_set_categories,
                        'enc_feat': self.enc_feat,
                        'dec_feat': self.dec_feat,
                        'seq_len_pred': self.seq_len_pred,
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'dropout': self.dropout
                      }
                    }
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path : str, dropout : Optional[float] = None):
        """
        Load the stored model at path
        """
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        if dropout is not None:
            checkpoint['kwargs']['dropout'] = dropout
        model = DropoutUncertaintyEncoderDecoderLSTM(**checkpoint['kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model