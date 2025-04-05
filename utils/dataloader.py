import torch
from torch.utils.data import Dataset
import numpy as np
from config.config import config
from sklearn.preprocessing import StandardScaler

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=config.SEQ_LEN):
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(data)
        self.X, self.y = make_sequences(self.normalized_data, seq_len)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])