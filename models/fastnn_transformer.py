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

        # fast_nn
        x1, x2 = self.fast_nn(x, is_training)
        combined = torch.cat([x1, x2], dim=-1)
        
        # transformer
        output = self.transformer(combined)

        return output
    
    def regularization_loss(self, model, tau, penalize_weights=config.PENALIZE_WEIGHTS):
        '''
        Regularization loss

        model: model to compute regularization loss for
        tau: tau value (parameter for regularization loss)
        penalize_weights: whether to penalize weights
        '''
        l1_penalty = torch.abs(self.fast_nn.variable_selection.weight) / tau # get the l1 penalty
        clipped_l1 = torch.clamp(l1_penalty, max=1.0) # caps penalty values at 1.0

        if penalize_weights:
            for name, param in model.named_parameters(): # this is probably really bad for transformer lol (update: transformer loves sparsity for some reason 😸)
                if len(param.shape) > 1: clipped_l1 += 0.001 * torch.sum(torch.abs(param))
        
        return torch.sum(clipped_l1) # return the sum of the clipped l1 penalty (add this to main loss)