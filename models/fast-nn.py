import torch
from torch import nn
import numpy as np
from config.config import config

class FactorAugmentedSparseThroughput(nn.Module):
    def __init__(self, input_dim, r_bar, depth, width, dp_mat, sparsity=None, rs_mat=None):
        super().__init__()
        p = input_dim # wanted to name it input_dim for consistency with the repo, paper refers to it as p
        self.diversified_projection = nn.Linear(p, r_bar, bias=False) # initialize linear projection for diversified projection layer
        dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32) # converts pretrained dp_mat to pytorch tensor
        self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=False) # assigns pretrained dp_mat to the linear layer (grad false since this layer is frozen)

        self.variable_selection = nn.Linear(p, width, bias=False) # initialize linear projection for variable selection layer
        
    def forward(self, x, is_training=False):
        x1 = self.diversified_projection(x)
        x2 = self.variable_selection(x)
        return x1, x2

    def regularization_loss(self, model, tau, penalize_weights=config.PENALIZE_WEIGHTS):
        l1_penalty = torch.abs(self.variable_selection.weight) / tau # get the l1 penalty
        clipped_l1 = torch.clamp(l1_penalty, max=1.0) # caps penalty values at 1.0

        if penalize_weights:
            for name, param in model.named_parameters(): # this is probably really bad for transformer lol
                if len(param.shape) > 1: clipped_l1 += 0.001 * torch.sum(torch.abs(param))
        
        return torch.sum(clipped_l1) # return the sum of the clipped l1 penalty (add this to main loss)