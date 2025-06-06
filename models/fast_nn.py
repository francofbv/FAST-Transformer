import torch
from torch import nn
import numpy as np
from config.config import config
'''
fast-nn model, partially taken from the original FAST-NN repo with some modifications

https://github.com/zhengli09/FAST-NN
'''

class FactorAugmentedSparseThroughput(nn.Module):
    '''
    Factor Augmented Sparse Throughput

    input_dim: input dimension
    r_bar: number of eigenvalues to keep
    width: width of the fast-nn model
    dp_mat: diversified projection matrix (pretrained)
    sparsity: sparsity of the fast-nn model
    rs_mat: random sparse matrix (for fast-nn model)
    '''
    def __init__(self, input_dim, r_bar, width, dp_mat, sparsity=None, rs_mat=None):
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