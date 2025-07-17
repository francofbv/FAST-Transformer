import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config.config import config

'''
basic time series transformer model implementation
'''

class TimeSeriesTransformer(nn.Module):
    '''
    Time Series Transformer for multi-step forecasting

    input_dim: input dimension
    d_model: model dimension
    nhead: number of attention heads
    num_layers: number of transformer layers
    output_dim: prediction horizon length
    '''
    def __init__(self, input_dim, d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS, output_dim=96):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model) # project into models dimension space
        self.pos_embedding = nn.Parameter(torch.randn(config.SEQ_LEN, d_model)) # learnable position embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True) # transformer layer w/ multi-head attention, FFN, normalization, residual connections
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # stack n_layer encoder layers
        self.output_proj = nn.Linear(d_model, output_dim) # output dimension = prediction horizon * num_variables

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        x = self.input_proj(x) + self.pos_embedding # add positional embeddings
        
        # No need to permute since we're using batch_first=True
        x = self.transformer(x)
        
        # Use the last timestep to predict the entire horizon
        x = x[:, -1] # (batch_size, d_model)
        x = self.output_proj(x) # (batch_size, output_dim) = (batch_size, pred_len * num_variables)

        return x