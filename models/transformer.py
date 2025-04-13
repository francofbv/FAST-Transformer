import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config.config import config

class TimeSeriesTransformer(nn.Module):
    '''
    Time Series Transformer

    input_dim: input dimension
    d_model: model dimension
    nhead: number of attention heads
    num_layers: number of transformer layers
    '''
    def __init__(self, input_dim, d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model) # project into models dimension space
        self.pos_embedding = nn.Parameter(torch.randn(config.SEQ_LEN, d_model)) # learnable position embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) # transformer layer w/ multi-head attention, FFN, normalization, residual connections
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # stack n_layer encoder layers
        self.output_proj = nn.Linear(d_model, 1) # output to 1 dimension

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        x = self.input_proj(x) + self.pos_embedding # add positional embeddings
        
        x = x.permute(1, 0, 2) # (seq_len, batch_size, d_model)
        x = self.transformer(x)
        
        x = x[-1] # (batch_size, d_model)
        x = self.output_proj(x)

        return x