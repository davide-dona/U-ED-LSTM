"""
Loss functions incorporating combined epistemic and aleatoric uncertainty estimations.

Based on:
- Kendall, Alex, and Yarin Gal. "What uncertainties do we need in bayesian deep learning for computer vision?." Advances in neural information processing systems 30 (2017).
- Weytjens, Hans, and Jochen De Weerdt. "Learning uncertainty with artificial neural networks for predictive process monitoring." Applied Soft Computing 125 (2022).
"""

import torch

class Loss:
    def __init__(self):
        pass
        
    def standard_mse(self, preds, targets):
        """
        Standard MSE loss.
        
        INPUTS:
        - preds: Predicted (time) values for N events: dim: seq len x batch x features (1)
        - targets: Target (time) values for N events: dim: batch x seq len
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        
        # Bring into structure: batch x seq len x output features (1)
        preds = preds.permute(1,0,2)
        targets = targets.unsqueeze(2)
        
        # Loss value
        L = torch.sum((targets - preds) ** 2, dim=2)
        # Mean over events in sequence
        L = torch.mean(L, dim=1)
        # Mean over batches
        L = torch.mean(L)
        
        return L
        
    def loss_attenuation_mse(self, pred_means, pred_logvars, targets):
        """
        Loss attenuation MSE: Combined Epistemic and Aleatoric Uncertainty.
    
        INPUTS:
        - pred_means: Predicted values for N events: dim: seq len x batch x output values (1)
        - pred_logvars: Predicted log variance values for predicted mean for N events: dim: seq len x batch x output values (1)
        - targets: Target (time) values for N events: dim: batch x sequence length
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        
        # Clamp the predicted variance to avoid extreme values
        min_logvariance = (torch.tensor(-6).to(pred_logvars.device))
        max_logvariance = (torch.tensor(6).to(pred_logvars.device))
        pred_logvars = torch.clamp(pred_logvars, min=min_logvariance, max=max_logvariance)
        
        # Bring into structure: batch x seq len x output features (1)
        pred_means = pred_means.permute(1,0,2)
        pred_logvars = pred_logvars.permute(1,0,2)
        targets = targets.unsqueeze(2)
        
        # Stable inverse variance: exp(-log(var)) = 1/sig^2 (1/var)
        inv_variances = torch.exp(-pred_logvars)
        
        L = torch.sum(0.5 * (inv_variances * ((targets - pred_means) ** 2) + pred_logvars), dim=2)
        # Mean over events in sequence
        L = torch.mean(L, dim=1)
        # Mean over batches
        L = torch.mean(L)

        return L
    
    def loss_attenuation_mse_log_normal(self, pred_means, pred_logvars, log_targets):
        """
        Loss attenuation MSE: Combined Epistemic and Aleatoric Uncertainty of an assumed Log normal probability density function for our time input.
    
        INPUTS:
        - pred_logmeans: Predicted log time values for N events: dim: seq len x batch x output values (1)
        - pred_logvars: Predicted log variance values for predicted mean for N events: dim: seq len x batch x output values (1)
        - targets: Target (time) values for N events: dim: batch x sequence length
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        
        # Clamp the predicted variance to avoid extreme values
        min_logvariance = (torch.tensor(-6).to(pred_logvars.device))
        max_logvariance = (torch.tensor(6).to(pred_logvars.device))
        pred_logvars = torch.clamp(pred_logvars, min=min_logvariance, max=max_logvariance)
                
        # Bring into structure: batch x seq len x output features (1)
        pred_means = pred_means.permute(1,0,2) # t := log(x)
        pred_logvars = pred_logvars.permute(1,0,2) # s := log(sigma^2)
        
        # Stable inverse variance: exp(-log(var)) = 1/sig^2 (1/var)
        inv_variances = torch.exp(-pred_logvars)
        
        # log the observed targets according to NLL log-normal PDF
        log_targets = log_targets.unsqueeze(2)
        
        L = torch.sum(log_targets + 0.5 * (pred_logvars + (inv_variances * (log_targets - pred_means)**2)), dim=2)
                
        # Mean over events in sequence
        L = torch.mean(L, dim=1)
        # Mean over batches
        L = torch.mean(L)

        return L
    
    def standard_cross_entropy(self, pred_logits, targets):
        """
        Standard Cross Entropy loss.
      
        INPUTS:
        - pred_logits: Predicted logit values for N events: dim: seq len x batch x labels (logit value for each label)
        - targets: Target class indices for N events: dim: batch x seq len
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """

        # Cross Entropy Loss
        CEL = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Change the shape of the prediction to: shape: batch_size x num_classes x seq len
        pred_logits = pred_logits.permute(1,2,0)
        
        L = CEL(input=pred_logits, target=targets)
        # Mean over events in sequences
        L = torch.mean(L, dim=1)
        # Mean over batches
        L = torch.mean(L) 
        
        return L
    
    def loss_attenuation_cross_entropy(self, pred_logits, pred_logvars, T, targets):
        """
        Loss attenuation cross entropy: Combined Epistemic and Aleatoric Uncertainty.
          
        INPUTS:
        - pred_logits: Predicted logit values for N events: dim: seq_len x batch x classes
        - pred_logvars: Predicted log variances per logit value for N events: dim: seq len x batch x classes
        - T: T gaussian distributed random epsilon value generations.
        - targets: Target class indices for N events: dim: batch x  seq len
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
            
        # Cross Entropy Loss
        CEL = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Get standard deviation
        variance = torch.exp(pred_logvars)
        std = torch.sqrt(variance)
        
        L = 0
        # T monte carlo iterations for approx. gaussian distribution
        for t in range(T):
            # epsilon_t: Generate a random matrix to distribute the standard deviations
            noise = torch.randn_like(pred_logits)    
            pred_logits_std_noise = pred_logits + std * noise
            
            # Change the shape of the prediction to: shape: batch_size x num_classes x seq len
            pred_logits_std_noise = pred_logits_std_noise.permute(1,2,0)
            
            # CEL of gaussian distributed unaries and target
            ce_loss = CEL(input=pred_logits_std_noise, target=targets)
            
            L += ce_loss
               
        L = (1/T) * L
        # Mean over events in sequence
        L = torch.mean(L, dim=1)
        # Mean over batches
        L = torch.mean(L)
          
        return L
        
