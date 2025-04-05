import torch
import torch.nn as nn
import numpy as np
from config.config import config
from .fast_nn import FactorAugmentedSparseThroughput
from .transformer import TimeSeriesTransformer

class FastNNTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS, r_bar=config.R_BAR, depth=config.DEPTH, width=config.WIDTH, dp_mat=None, sparsity=None, rs_mat=None):
        super().__init__()
        
        self.fast_nn = FactorAugmentedSparseThroughput(
            input_dim=input_dim,
            r_bar=r_bar,
            depth=depth,
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

        # fast_nn
        x1, x2 = self.fast_nn(x, is_training)
        combined = torch.cat([x1, x2], dim=-1)
        
        # transformer
        output = self.transformer(combined)

        return output
        
    def regularization_loss(self, tau):
        return self.fast_nn.regularization_loss(self.transformer, tau)