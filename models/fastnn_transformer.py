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
    def __init__(self, dp_mat, input_dim=config.INPUT_DIM, d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS, r_bar=config.R_BAR, width=config.WIDTH, sparsity=None, rs_mat=None):
        super().__init__()
        
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
            num_layers=num_layers
        )
        
    def forward(self, x, is_training=False):
        '''
        Forward pass

        x: input data
        is_training: whether the model is in training mode
        '''
        # Reshape input: (batch_size, seq_len, num_stocks, num_features) -> (batch_size * seq_len * num_stocks, num_features)
        batch_size, seq_len, num_stocks, num_features = x.shape
        x_reshaped = x.reshape(-1, num_features)
        
        # fast_nn
        x1, x2 = self.fast_nn(x_reshaped, is_training)
        
        # Reshape back: (batch_size * seq_len * num_stocks, r_bar/width) -> (batch_size, seq_len, num_stocks, r_bar/width)
        x1 = x1.reshape(batch_size, seq_len, num_stocks, -1)
        x2 = x2.reshape(batch_size, seq_len, num_stocks, -1)
        
        # Combine features
        combined = torch.cat([x1, x2], dim=-1)
        
        # Reshape for transformer: (batch_size, seq_len, num_stocks, input_dim) -> (batch_size * num_stocks, seq_len, input_dim)
        combined = combined.reshape(batch_size * num_stocks, seq_len, -1)
        
        # transformer
        output = self.transformer(combined)
        
        # Reshape output back: (batch_size * num_stocks, 1) -> (batch_size, num_stocks)
        output = output.reshape(batch_size, num_stocks)

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