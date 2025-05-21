import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from config.config import config

class ETTh1Dataset(Dataset):
    def __init__(self, data_path, seq_len=config.SEQ_LEN, is_training=True):
        '''
        Initialize ETTh1 dataset
        
        Args:
            data_path: Path to the CSV file
            seq_len: Length of the sequence for the transformer
            is_training: Whether this is training data (affects target calculation)
        '''
        self.seq_len = seq_len
        self.is_training = is_training

        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        