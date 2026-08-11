"""
Comprehensive efficient auto-regressive model training (with epistemic and aleatoric uncertainties).

Uses gradient normalization (GradNorm) technique to balance task losses dynamically based on gradient magnitudes:
- Chen Z. et.al, GradNorm: "Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", ICML, 2018.
"""

import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np

class Trainer:
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 loss_obj,
                 log_normal_loss_num_feature,
                 optimize_values,
                 suffix_data_split_value,
                 writer,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'model.pkl',
                 gradnorm_values=None):
        """
        Trainer class constructor.
        
        ARGS:
        - device: Device (GPU or CPU).
        - model: Model that is trained and validated.
        - data_train: Training data.
        - data_val: Validation data.
        - loss_obj: object for loss functions
        - log_normal_loss_num_feature: list of strings of num feaures that follow log normal distribution.
        - optimize_values:
            - regularization_term: L2 regularization for weights, biases, and dropout of stochastic model. 
            - optimizer: Optimization algorithm for training.
            - epochs: Epochs the model trains the full training dataset.
            - mini_batches: Batches the model get passed at once.
            - shuffles: Shuffle batches.
            - teacher_forcing_ratio: Value [0,1) that is used to decide if predicted or next target event is used for next prediction by model.
            - gradnorm: 
        
        - suffix_data_split_value: Number of last values of suffix events. 
        - writer
        - save_model_n_th_epoch: int,
        - saving_path: str, default: 'model.pkl'
        
        - gradnorm_values:
            - use_gradnorm: Boolean value if gradnorm should be used or not.
            - gn_alpha": Hyperparameter for GradNorm.
            - gn_learning_rate: Task balancing weights leraning rate.
            - gn_weights: Weights for losse: Initial None.
            - T: Initial sum of weights: Initial 0
            - l0: Initial loss: Initial None
            - gn_optimizer: Optimizer to determine new weights for losses: Initial None  
        """

        # Standard Training parameters
        self.device = device
        print("Device: ", device)
        self.model = model.to(device)
        print("Model: ", model)
        self.data_train = data_train
        print("Train Dataset: ", data_train)
        self.data_val = data_val
        print("Validation Dataset: ", data_val)
        
        self.loss_obj = loss_obj
        print("Loss object for method calling: ", loss_obj)
        self.log_normal_loss_num_feature = log_normal_loss_num_feature
        print("Num. feautures that follow log-normal PDF: ", log_normal_loss_num_feature)
        
        # Standard Optimization parameters
        self.optimize_values = optimize_values
        self.regularization_term = optimize_values["regularization_term"]
        print("regularization: ", self.regularization_term)
        self.optimizer = optimize_values["optimizer"]
        print("Optimizer: ", self.optimizer)
        self.scheduler = optimize_values["scheduler"]
        print("Scheduler: ", self.scheduler)
        self.epochs = optimize_values["epochs"]
        print("Epochs: ", self.epochs)
        self.mini_batches = optimize_values["mini_batches"]
        print("Mini baches: ", self.mini_batches)
        self.shuffle = optimize_values["shuffle"]
        print("Shuffle batched dataset: ", self.shuffle)
        
        # Teacher forcing
        self.teacher_forcing_ratio = optimize_values["teacher_forcing_ratio"]
        print("Teacher forcing ratio: ", self.teacher_forcing_ratio)
        self.teacher_forcing_init = float(self.teacher_forcing_ratio)  # e.g. 1.0
        
        # Events in sufffix: Dependent on data set
        self.suffix_data_split_value = suffix_data_split_value
        
        # TensorBoard
        self.writer = writer
        
        # Model saving
        self.save_model_n_th_epoch = save_model_n_th_epoch
        self.saving_path = saving_path

        # Gradnorm parameters
        self.gradnorm_values = gradnorm_values
        if gradnorm_values is not None:
            self.use_gradnorm = gradnorm_values["use_gradnorm"]
            print("Use GradNorm: ", self.use_gradnorm)
            
            if self.use_gradnorm:
                self.gn_alpha = gradnorm_values["gn_alpha"]
                print("GradNorm alpha: ", self.gn_alpha)
                self.gn_learning_rate = gradnorm_values["gn_learning_rate"]
                print("GradNorm learning rate: ", self.gn_learning_rate)
                self.number_tasks = gradnorm_values["number_tasks"]
                
                self.gn_weights = torch.nn.Parameter(torch.ones(self.number_tasks, dtype=torch.float))
                print("Initial GradNorm loss weights: ",self.gn_weights)
                self.l0 = None
                print("Initial loss values: ", self.l0)
                
                # Use same optimizer as for the model optimization
                self.gn_optimizer = torch.optim.AdamW(params=[self.gn_weights], lr=self.gn_learning_rate)
                print("GradNorm optimizer: ", self.gn_optimizer)
    
    def train_model(self):
        """
        Seq2Seq Multi Task Learning algorithm with uncertainties.
        
        Returns:
        - train_attenuated_losses:
        - val_losses:
        - val_attenuated_losses
        """
        # Train the model
        self.model.train()

        # Lists to store the losses
        train_attenuated_losses = []
        val_losses = []
        val_attenuated_losses = []

        # Validation dataloader
        val_dataloader = DataLoader(dataset=self.data_val, batch_size=self.mini_batches, shuffle=self.shuffle, num_workers=4, pin_memory=True)
        
        # Teacher forcing reducing index:
        k = 1

        # choose sigmoid teacher forcing hyperparams (tweak to taste)
        midpoint = 0.5 * self.epochs          # default: halfway through training
        scale = 0.1 * self.epochs             # default: transition width (10% of total epochs)
        
        # Trainings/ Epoch Loop
        for epoch in tqdm(range(self.epochs)):
            
            # Train dataloader
            train_dataloader = DataLoader(dataset=self.data_train, batch_size=self.mini_batches, shuffle=self.shuffle, num_workers=4, pin_memory=True)
            
            epoch_cat_loss = {}
            epoch_num_loss = {}

            epoch_loss = 0.0
            num_batches_per_epoch = 0.0
            
            # Reduce Teacher forcing ratio dynamically:
            if epoch >= ((self.epochs * k) / 5):
                # adopt to reduce teacher forcing more drastical:
                self.teacher_forcing_ratio = self.teacher_forcing_ratio - (self.teacher_forcing_ratio / 10)
                if self.teacher_forcing_ratio < 0:
                    self.teacher_forcing_ratio = 0.0
                k +=1

            # Sigmoid decreasing schedule
            # Starts near 1, smoothly decays to near 0
            # self.teacher_forcing_ratio = 1 / (1 + np.exp((epoch - midpoint) / scale))

            # Bacth Loop
            for i, train_data in enumerate(train_dataloader): 
                cats, nums, _ = train_data
                            
                # Automize this            
                
                            
                # dim: list(list(Tensors categorical: dim: batch size x window size-4), list(Tensors numerical: dim: batch size x window size-4))
                prefixes_cat = [cat[:, :-self.suffix_data_split_value].to(self.device) for cat in cats]
                prefixes_num = [num[:, :-self.suffix_data_split_value].to(self.device) for num in nums]
                prefixes = [prefixes_cat, prefixes_num]
                
                suffixes_cat = [cat[:, -self.suffix_data_split_value:].to(self.device) for cat in cats]
                suffixes_num = [num[:, -self.suffix_data_split_value:].to(self.device) for num in nums]
                suffixes = [suffixes_cat, suffixes_num]
                
                # GradNorm Training:
                if self.use_gradnorm:
                    # all_losses: list of two dicts, categorical dict and numerical dict: key: feature name, value: tensor loss
                    # loss: Tensor of total loss
                    all_losses_dict, loss_value = self.train_epoch_gradnorm(prefixes=prefixes, suffixes=suffixes)
                    cat_losses_dict, num_losses_dict = all_losses_dict
                     
                # Standard Training:    
                else:
                    all_losses_dict, loss_value = self.train_epoch(prefixes=prefixes, suffixes=suffixes)
                    cat_losses_dict, num_losses_dict = all_losses_dict
                
                # Accumulate the categorical losses
                for feature_name in cat_losses_dict.keys():  
                    if feature_name in epoch_cat_loss:
                        # Add the current batch's loss to the cumulative loss
                        epoch_cat_loss[feature_name] += cat_losses_dict[feature_name].item()
                    else:
                        # Initialize the cumulative loss with the first batch's loss
                        epoch_cat_loss[feature_name] = cat_losses_dict[feature_name].item()

                # Accumulate the numerical losses
                for feature_name in num_losses_dict.keys():  
                    if feature_name in epoch_num_loss:
                        # Add the current batch's loss to the cumulative loss
                        epoch_num_loss[feature_name] += num_losses_dict[feature_name].item()
                    else:
                        # Initialize the cumulative loss with the first batch's loss
                        epoch_num_loss[feature_name] = num_losses_dict[feature_name].item()

                # Accumulated total loss for the entire epoch
                epoch_loss += loss_value.item()
                                
                # Increase number of trained batches:
                num_batches_per_epoch += 1
                
            # Take the mean losses over all batches
            for feature_name in epoch_cat_loss.keys():
                epoch_cat_loss[feature_name] = epoch_cat_loss[feature_name] / num_batches_per_epoch
                
            for feature_name in epoch_num_loss.keys():
                epoch_num_loss[feature_name] = epoch_num_loss[feature_name] / num_batches_per_epoch
            
            epoch_loss_train = epoch_loss / num_batches_per_epoch

            # Current learning rate
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            
            # Prints per Epoch:
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {current_lr}, Teacher forcing ratio: {self.teacher_forcing_ratio}")
            
            tqdm.write(f"Training: Avg Attenuated Training Loss: {epoch_loss_train:.4f}")
            
            train_attenuated_losses.append(epoch_loss_train)
            
            # Validation
            epoch_cat_loss_val_std, epoch_cat_loss_val_unc, epoch_num_loss_val_std, epoch_num_loss_val_unc, epoch_loss_val_std, epoch_loss_val_unc = self.validation_epoch(val_dataloader=val_dataloader)
                        
            tqdm.write(f"Validation: Avg Standard Validation Loss: {epoch_loss_val_std:.4f}")
            tqdm.write(f"Validation: Avg Attenuated Validation Loss: {epoch_loss_val_unc:.4f}")
        
            val_losses.append(epoch_loss_val_std)
            val_attenuated_losses.append(epoch_loss_val_unc)

            # Tensorboard writer:
            # Hyperparameters
            self.writer.add_scalars(
                "Hyperparameter:", 
                {
                    'Learning Rate': current_lr,
                    'Teacher Forcing Ratio': self.teacher_forcing_ratio
                },
                epoch+1)
            
            # Total losses
            self.writer.add_scalars(
                "Total Losses", 
                {
                    'Training Total': epoch_loss_train,
                    'Standard Validation Total': epoch_loss_val_std,
                    'Uncertainty Validation Total': epoch_loss_val_unc
                    },
                epoch+1)
            
            # Categorical losses
            for feature_name in epoch_cat_loss.keys():
                self.writer.add_scalars(
                    "Categorical Feature Losses",
                    {
                        f'Training {feature_name}': epoch_cat_loss[feature_name],
                        f'Standard Validation {feature_name}': epoch_cat_loss_val_std[feature_name],
                        f'Uncertainty Validation {feature_name}': epoch_cat_loss_val_unc[feature_name]
                    },
                    epoch + 1)
                  
            # Numerical losses
            for feature_name in epoch_num_loss.keys():
                self.writer.add_scalars(
                    "Numerical Feature Losses",
                    {
                        f'Training {feature_name}': epoch_num_loss[feature_name],
                        f'Standard Validation {feature_name}': epoch_num_loss_val_std[feature_name],
                        f'Uncertainty Validation {feature_name}': epoch_num_loss_val_unc[feature_name]
                    },
                    epoch + 1)
                
            # GradNorm
            if self.use_gradnorm:
                # Convert the GradNorm weights and gradient norms to numpy arrays
                write_weights = self.gn_weights.data.cpu().numpy()
                
                feature_losses = list(epoch_cat_loss.keys()) + list(epoch_num_loss.keys())
                for i, feature_name in enumerate(feature_losses):
                    self.writer.add_scalars(
                        "Gradnorm values", 
                        {
                            f'Gradnorm Weight {feature_name}': write_weights[i]
                        },
                        epoch+1)
            
            # Adjust the learning rate if necessary
            tqdm.write(f"Validation Loss for Scheduler: {epoch_loss_val_std:.4f}")
            
            # Adjust learning rate
            self.scheduler.step(epoch_loss_val_std)

            if (i+1) % self.save_model_n_th_epoch == 0:
                 tqdm.write("saving model")
                 self.model.save(self.saving_path)
                                 
        print("Training complete.")

        self.model.save(self.saving_path)
        tqdm.write(f'Model saved to path: {self.saving_path}')

        return train_attenuated_losses, val_losses, val_attenuated_losses
    
    def train_epoch(self, prefixes, suffixes):
        """
        Train the model on batches.

        INPUTS:
        - prefixes: 
        - suffixes: 

        OUTPUTS:
        - all_losses:
        - loss: 
        """
        # predictions: List of two Dicts one for categorical (means and vars), one for numerical (means and vars): key: feature name + _mean or _var, value: tensor with dim: seq len x batch size x output feature size
        # data_features_indeces_dec: List of two Dicts one for categorical, one for numerical: key: feature name, value: index of tensor in data list
        predictions, _, _, data_features_indeces_dec= self.model(prefixes=prefixes, suffixes=suffixes, teacher_forcing_ratio=self.teacher_forcing_ratio)
        
        # Get cat and num predictions
        predictions_cat, predictions_num = predictions
        
        # cat, num feature index dict
        cat_features_indeces, num_features_indeces = data_features_indeces_dec
        
        # Get cat and num targets
        cat_suffixes, num_suffixes = suffixes
        
        cat_suffixes_dict = {}
        # For suffix: map the feature name of the decoder output to the corresponding tensor using the index
        for feature_name, index in cat_features_indeces.items():
            cat_suffixes_dict[feature_name] = cat_suffixes[index]
            
        num_suffixes_dict = {}    
        for feature_name, index in num_features_indeces.items():
            num_suffixes_dict[feature_name] = num_suffixes[index]        
        
        # Calculate the loss for all categorical features
        cat_loss_dict = {}
        cat_loss_list = []
        # Iterate over the dictionary and group by feature name without suffix
        for key in predictions_cat:
            if key.endswith('_mean'):
                # Get the base name of the feature (removing the '_mean' suffix)
                feature_name = key[:-5]

                # Get the corresponding mean and variance tensors from the prediction
                mean_cat_pred = predictions_cat[key] # dim: seq len x batch size x number classes
                var_cat_pred = predictions_cat.get(f'{feature_name}_var') # dim: seq len x batch size x number classes
                target_cat = cat_suffixes_dict[feature_name] # dim: batch size x seq len
                # Loss caluclation
                loss_cat = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred, pred_logvars=var_cat_pred, T=30, targets=target_cat.long())
                
                if (feature_name in cat_loss_dict):
                    raise ValueError("Feature is already in output dict")
                
                cat_loss_dict[feature_name] = loss_cat
                cat_loss_list.append(loss_cat)
        
        # Calculate the loss for all numerical features
        num_loss_dict = {}
        num_loss_list = []
        # Iterate over the dictionary and group by feature name without suffix
        for key in predictions_num:
            if key.endswith('_mean'):
                # Get the base name of the feature (removing the '_mean' suffix)
                feature_name = key[:-5]
                            
                # Get the corresponding mean and variance tensors from the prediction
                mean_num_pred = predictions_num[key] # dim: seq len x batch size x number classes
                var_num_pred = predictions_num.get(f'{feature_name}_var') # dim: seq len x batch size x 1
                target_num = num_suffixes_dict[feature_name] # dim: batch size x seq len

                # Normal loss:
                loss_num = self.loss_obj.loss_attenuation_mse(pred_means=mean_num_pred, pred_logvars=var_num_pred, targets=target_num)
                                
                if (feature_name in num_loss_dict):
                    raise ValueError("Feature is already in output dict")

                num_loss_dict[feature_name] = loss_num
        
        # List of dictionary of categorical and numerical losses for 1 batch
        all_losses = [cat_loss_dict, num_loss_dict]
        
        weight_reg_enc, bias_reg_enc = self.model.encoder.regularizer()
        weight_reg_dec, bias_reg_dec = self.model.decoder.regularizer()
        
        weight_reg = weight_reg_enc + weight_reg_dec
        bias_reg = bias_reg_enc + bias_reg_dec

        # Zero gradients before optimization step
        self.optimizer.zero_grad()

        # Total mean loss
        stacked_tensor_losses = torch.stack((cat_loss_list + num_loss_list))
        data_loss = stacked_tensor_losses.sum()
        loss = data_loss + self.regularization_term * (weight_reg.to(self.device) + bias_reg.to(self.device))
        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
        # Model Optimization
        self.optimizer.step()
        
        return all_losses, loss

    def train_epoch_gradnorm(self, prefixes, suffixes):
        """
        Train the model on batches and weight MTL lossses using GradNorm.

        INPUTS:
        - prefixes: list(categorical list(tensors: batch_size x window_size - suffix_size), numerical list(tensors: batch_size x window_size - suffix_size))
        - suffixes:  list(categorical list(tensors: batch_size x suffix_size - end), numerical list(tensors: batch_size x suffix_size - end))

        OUTPUTS:
        - weighted_loss: weighted GradNorm loss 
        - loss: Total loss: weighted_loss + regularizations
        """
        # Predictions  
        predictions, (h,_), _, data_features_indeces_dec= self.model(prefixes=prefixes, suffixes=suffixes, teacher_forcing_ratio=self.teacher_forcing_ratio)
        
        predictions_cat, predictions_num = predictions
        
        # Targets
        cat_features_indeces, num_features_indeces = data_features_indeces_dec
        
        cat_suffixes, num_suffixes = suffixes
        
        cat_suffixes_dict = {}    
        for feature_name, index in cat_features_indeces.items():
            cat_suffixes_dict[feature_name] = cat_suffixes[index]
        
        num_suffixes_dict = {}
        for feature_name, index in num_features_indeces.items():
            num_suffixes_dict[feature_name] = num_suffixes[index]    
        
        # Caluclate the loss for all categorical features
        cat_loss_dict = {}
        cat_loss_list = []
        for key in predictions_cat:
            if key.endswith('_mean'):
                # Get the base name of the feature (removing the '_mean' suffix)
                feature_name = key[:-5]

                # Get the corresponding mean and variance tensors from the prediction
                mean_cat_pred = predictions_cat[key] # dim: seq len x batch size x number classes
                var_cat_pred = predictions_cat.get(f'{feature_name}_var') # dim: seq len x batch size x number classes
                target_cat = cat_suffixes_dict[feature_name] # dim: batch size x seq len

                # Loss caluclation
                loss_cat = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred, pred_logvars=var_cat_pred, T=30, targets=target_cat.long())

                if (feature_name in cat_loss_dict):
                    raise ValueError("Feature is already in output dict")

                cat_loss_dict[feature_name] = loss_cat
                cat_loss_list.append(loss_cat)
        
        # Caluclate the loss for all numerical features
        num_loss_dict = {}
        num_loss_list = []
        for key in predictions_num:
            if key.endswith('_mean'):
                # Get the base name of the feature (removing the '_mean' suffix)
                feature_name = key[:-5]
                # print(feature_name)
                            
                # Get the corresponding mean and variance tensors from the prediction
                mean_num_pred = predictions_num[key] # dim: seq len x batch size x number classes
                var_num_pred = predictions_num.get(f'{feature_name}_var') # dim: seq len x batch size x 1
                target_num = num_suffixes_dict[feature_name] # dim: batch size x seq len

                # Normal loss:
                loss_num = self.loss_obj.loss_attenuation_mse(pred_means=mean_num_pred, pred_logvars=var_num_pred, targets=target_num)
                # print("normal loss")
                                
                if (feature_name in num_loss_dict):
                    raise ValueError("Feature is already in output dict")

                num_loss_dict[feature_name] = loss_num
                num_loss_list.append(loss_num)
        
        all_losses = [cat_loss_dict, num_loss_dict]
           
        # Initialize GradNorm variables at the first iteration
        all_loss_list = (cat_loss_list + num_loss_list)
        
        gn_seq_loss = torch.stack((cat_loss_list + num_loss_list))
        
        if self.l0 is None:            
            # Initial loss for each task
            self.l0 = torch.stack([loss.detach() for loss in all_loss_list])  # Detach to prevent gradients for initial losses
        
        self.optimizer.zero_grad()

        weight_reg_enc, bias_reg_enc = self.model.encoder.regularizer()
        weight_reg_dec, bias_reg_dec = self.model.decoder.regularizer()

        weight_reg = weight_reg_enc + weight_reg_dec
        bias_reg = bias_reg_enc + bias_reg_dec

        # Ensure that the GradNorm weights are on the correct device
        weighted_loss = self.gn_weights.to(self.device) * gn_seq_loss
        loss = weighted_loss.sum() + self.regularization_term * (weight_reg.to(self.device) + bias_reg.to(self.device))

        # Compute gradients for model parameters
        loss.backward(retain_graph=True)

        # Weight updating using GradNorm                   
        gw = [torch.norm(torch.autograd.grad((self.gn_weights[i] * gn_loss), h, retain_graph=True, create_graph=True)[0]) for i, gn_loss in enumerate(gn_seq_loss)]
        gw_all = torch.stack(gw)

        # Compute the average gradient norm
        gw_all_avg = gw_all.mean().detach()
        # Compute loss ratio per task
        loss_ratio = (gn_seq_loss / self.l0).detach()
        # Compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()

        # Compute the GradNorm loss
        constant = (gw_all_avg * rt ** self.gn_alpha).detach()
        gradnorm_loss = torch.abs(gw_all - constant).sum()
        
        # clear gradients of weights
        self.gn_optimizer.zero_grad()
        # backward pass for GradNorm
        gradnorm_loss.backward()

        # Optimize loss weights
        self.gn_optimizer.step()
        
        # Renormalization of weights
        self.gn_weights.data = (self.gn_weights.data / self.gn_weights.data.sum() * self.number_tasks).detach()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimize model parameters
        self.optimizer.step()
        
        return all_losses, loss

    def validation_epoch(self, val_dataloader):
        """
        Validates the model on the validation set during training.

        INPUTS:
        - val_dataloader: Validation data for validating the model during training.
        
        OUTPUTS:
        - cat_loss_dict_std: Cat. event attributes standard loss
        - cat_loss_dict_unc: Cat. event attributes attenuated loss
        - num_loss_dict_std, Con. event attributes standard loss
        - num_loss_dict_unc: Con. event attributes attenuated loss
        - val_epoch_loss_std: Total event attributes standard loss
        - val_epoch_loss_unc: Total event attributes attenuated loss
        """
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
        
            cat_loss_dict_std = {}
            cat_loss_dict_unc = {}
            
            num_loss_dict_std = {}
            num_loss_dict_unc = {}
            
            num_batches_per_epoch = 0.0
            
            for _, val_data in enumerate(val_dataloader): 
                
                cats, nums, _ = val_data    
                
                # dim: list(list(Tensors categorical: dim: batch size x window size-4), list(Tensors numerical: dim: batch size x window size-4))
                prefixes_cat = [cat[:, :-self.suffix_data_split_value].to(self.device) for cat in cats]
                prefixes_num = [num[:, :-self.suffix_data_split_value].to(self.device) for num in nums]
                prefixes = [prefixes_cat, prefixes_num]
                    
                suffixes_cat = [cat[:, -self.suffix_data_split_value:].to(self.device) for cat in cats]
                suffixes_num = [num[:, -self.suffix_data_split_value:].to(self.device) for num in nums]
                suffixes = [suffixes_cat, suffixes_num]

                # Model predictions:
                predictions, _, _, data_features_indeces_dec= self.model(prefixes=prefixes, suffixes=suffixes, teacher_forcing_ratio=self.teacher_forcing_ratio)
                predictions_cat, predictions_num = predictions
                
                # Targets
                cat_features_indeces, num_features_indeces = data_features_indeces_dec
                
                cat_suffixes, num_suffixes = suffixes
                
                cat_suffixes_dict = {}
                num_suffixes_dict = {}
                
                for feature_name, index in cat_features_indeces.items():
                    cat_suffixes_dict[feature_name] = cat_suffixes[index]
                
                for feature_name, index in num_features_indeces.items():
                    num_suffixes_dict[feature_name] = num_suffixes[index]    
                
                for key in predictions_cat:
                    if key.endswith('_mean'):
                        # Get the base name of the feature (removing the '_mean' suffix)
                        feature_name = key[:-5]

                        # Get the corresponding mean and variance tensors from the prediction
                        mean_cat_pred = predictions_cat[key] # dim: seq len x batch size x number classes
                        var_cat_pred = predictions_cat.get(f'{feature_name}_var') # dim: seq len x batch size x number classes
                        target_cat = cat_suffixes_dict[feature_name] # dim: batch size x seq len

                        # Loss caluclation
                        # Standard cross entropy
                        cat_loss_std = self.loss_obj.standard_cross_entropy(pred_logits=mean_cat_pred, targets=target_cat.long())
                        # Uncertainty cross entropy
                        cat_loss_unc = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred, pred_logvars=var_cat_pred, T=30, targets=target_cat.long())

                        if feature_name in cat_loss_dict_std:
                            # Add the current batch's loss to the cumulative loss
                            cat_loss_dict_std[feature_name] += cat_loss_std
                            cat_loss_dict_unc[feature_name] += cat_loss_unc
                        else:
                            # Initialize the cumulative loss with the first batch's loss
                            cat_loss_dict_std[feature_name] = cat_loss_std.clone()
                            cat_loss_dict_unc[feature_name] = cat_loss_unc.clone()
                
                for key in predictions_num:
                    if key.endswith('_mean'):
                        # Get the base name of the feature (removing the '_mean' suffix)
                        feature_name = key[:-5]

                        # Get the corresponding mean and variance tensors from the prediction
                        mean_num_pred = predictions_num[key] # dim: seq len x batch size x number classes
                        var_num_pred = predictions_num.get(f'{feature_name}_var') # dim: seq len x batch size x 1
                        target_num = num_suffixes_dict[feature_name] # dim: batch size x seq len

                        # Loss caluclation
                        # Standard mean squared error
                        num_loss_std = self.loss_obj.standard_mse(preds=mean_num_pred, targets=target_num)
                        # Normal loss:
                        num_loss_unc = self.loss_obj.loss_attenuation_mse(pred_means=mean_num_pred, pred_logvars=var_num_pred, targets=target_num)
                        
                        if feature_name in num_loss_dict_std:
                            # Add the current batch's loss to the cumulative loss
                            num_loss_dict_std[feature_name] += num_loss_std
                            num_loss_dict_unc[feature_name] += num_loss_unc
                        else:
                            # Initialize the cumulative loss with the first batch's loss
                            num_loss_dict_std[feature_name] = num_loss_std.clone()
                            num_loss_dict_unc[feature_name] = num_loss_unc.clone()   
                
                # Increase number of trained batches:
                num_batches_per_epoch += 1
                
            # Average losses over batches
            for feature_name in cat_loss_dict_std.keys():
                cat_loss_dict_std[feature_name] /= num_batches_per_epoch
                cat_loss_dict_unc[feature_name] /= num_batches_per_epoch

            for feature_name in num_loss_dict_std.keys():
                num_loss_dict_std[feature_name] /= num_batches_per_epoch
                num_loss_dict_unc[feature_name] /= num_batches_per_epoch

            # Sum all feature-wise losses to get total epoch losses
            val_epoch_loss_std = (sum(cat_loss_dict_std.values()) + sum(num_loss_dict_std.values())).item()
            val_epoch_loss_unc = (sum(cat_loss_dict_unc.values()) + sum(num_loss_dict_unc.values())).item()
                
        # Set model back to train for gradient caluclation and optimization.
        self.model.train()
        
        return cat_loss_dict_std, cat_loss_dict_unc, num_loss_dict_std, num_loss_dict_unc, val_epoch_loss_std, val_epoch_loss_unc