import torch
import torch.nn as nn
import numpy as np
from config.config import config
from .fast_nn import FactorAugmentedSparseThroughput
from .transformer import TimeSeriesTransformer

class FastNNTransformer(nn.Module):
    '''
    Fast-NN Transformer combined model

    dp_mat: diversified projection matrix (pretrained)
    input_dim: input dimension
    d_model: model dimension
    nhead: number of attention heads
    num_layers: number of transformer layers
    r_bar: number of eigenvalues to keep
    width: width of the fast-nn model
    sparsity: sparsity of the fast-nn model
    rs_mat: random sparse matrix (for fast-nn model)
    '''
    def __init__(self, dp_mat, input_dim=config.INPUT_DIM, d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS, r_bar=config.R_BAR, width=config.WIDTH, pred_len=96, output_dim=1, sparsity=None, rs_mat=None):
        super().__init__()
        
        self.pred_len = pred_len
        self.output_dim = output_dim  # Number of variables to predict (1 for univariate, 7 for multivariate)
        
        self.fast_nn = FactorAugmentedSparseThroughput(
            input_dim=input_dim,
            r_bar=r_bar,
            width=width,
            dp_mat=dp_mat,
            sparsity=sparsity,
            rs_mat=rs_mat
        )

        self.transformer = TimeSeriesTransformer(
            input_dim=r_bar + width,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=pred_len * output_dim  # Multi-step, multi-variate forecasting output
        )
        
    def forward(self, x, is_training=False):
        '''
        Forward pass for ETTh1 time series forecasting

        x: input data of shape (batch_size, seq_len, num_features)
        is_training: whether the model is in training mode
        '''
        # For ETTh1: input shape is (batch_size, seq_len, num_features)
        batch_size, seq_len, num_features = x.shape
        
        # Reshape for FAST-NN: (batch_size, seq_len, num_features) -> (batch_size * seq_len, num_features)
        x_reshaped = x.reshape(-1, num_features)
        
        # Apply FAST-NN feature selection
        x1, x2 = self.fast_nn(x_reshaped, is_training)
        
        # Reshape back: (batch_size * seq_len, r_bar/width) -> (batch_size, seq_len, r_bar/width)
        x1 = x1.reshape(batch_size, seq_len, -1)
        x2 = x2.reshape(batch_size, seq_len, -1)
        
        # Combine FAST-NN outputs
        combined = torch.cat([x1, x2], dim=-1)  # (batch_size, seq_len, r_bar + width)
        
        # Apply transformer (now outputs pred_len * output_dim dimensions directly)
        output = self.transformer(combined)  # (batch_size, pred_len * output_dim)
        
        # Reshape to (batch_size, pred_len, output_dim) for multivariate forecasting
        output = output.view(output.shape[0], self.pred_len, self.output_dim)

        return output
    
    def regularization_loss(self, model, tau, penalize_weights=config.PENALIZE_WEIGHTS):
        '''
        Regularization loss

        model: model to compute regularization loss for
        tau: tau value (parameter for regularization loss)
        penalize_weights: whether to penalize weights
        '''
        # Only penalize the variable selection layer
        l1_penalty = torch.mean(torch.abs(self.fast_nn.variable_selection.weight)) / tau
        
        if penalize_weights:
            # Add small L2 regularization for other parameters
            l2_reg = 0.0
            for name, param in model.named_parameters():
                if 'variable_selection' not in name and len(param.shape) > 1:
                    l2_reg += torch.norm(param)
            l1_penalty += 0.001 * l2_reg
        
        return l1_penalty
    
    def predict_multi_horizon(self, x, horizons=[96, 192, 336, 720]):
        '''
        Predict multiple forecasting horizons for multivariate output
        
        x: input data of shape (batch_size, seq_len, num_features)
        horizons: list of prediction horizons
        '''
        results = {}
        
        # Use the model's current prediction length as base
        base_pred = self.forward(x)  # (batch_size, pred_len, output_dim)
        
        for horizon in horizons:
            if horizon <= self.pred_len:
                # Truncate if horizon is smaller than model's prediction length
                results[horizon] = base_pred[:, :horizon, :]
            else:
                # For longer horizons, we would need autoregressive prediction
                # For now, repeat the last prediction (this is a simple approach)
                # In practice, you might want to retrain models for each horizon
                extended = torch.cat([
                    base_pred,
                    base_pred[:, -1:, :].repeat(1, horizon - self.pred_len, 1)
                ], dim=1)
                results[horizon] = extended
                
        return results