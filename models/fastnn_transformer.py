import torch
import torch.nn as nn
import numpy as np
from config.config import config
from .fast_nn import FactorAugmentedSparseThroughput
from .transformer import TimeSeriesTransformer

class FastNNTransformer(nn.Module):
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
            #input_dim=r_bar + width,
            input_dim=width + input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
    def forward(self, x, is_training=False):

        # fast_nn
        x1, x2 = self.fast_nn(x, is_training)
        #print(x1.shape)
        #print(x2.shape)#
        #exit()
        combined = torch.cat([x1, x2], dim=-1)
        
        # transformer
        output = self.transformer(combined)

        return output
    
    def regularization_loss(self, model, tau, penalize_weights=config.PENALIZE_WEIGHTS):
        l1_penalty = torch.abs(self.fast_nn.variable_selection.weight) / tau # get the l1 penalty
        clipped_l1 = torch.clamp(l1_penalty, max=1.0) # caps penalty values at 1.0

        if penalize_weights:
            for name, param in model.named_parameters(): # this is probably really bad for transformer lol
                if len(param.shape) > 1: clipped_l1 += 0.001 * torch.sum(torch.abs(param))
        
        return torch.sum(clipped_l1) # return the sum of the clipped l1 penalty (add this to main loss)