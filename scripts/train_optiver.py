import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # too lazy to change the paths so this is a quicker fix lol
sys.path.append(project_root)

from config.config import config
from models.transformer import TimeSeriesTransformer
from models.fastnn_transformer import FastNNTransformer
from utils.optiver_dataloader import OptiverDataset
from scripts.evaluate import evaluate

'''
defines the model instantiation, first layer weight matrix, and primary training and validation loop
'''

def compute_dp_mat(x, r_bar=config.R_BAR):
    '''
    Compute the pretrained dp (diversified projection)matrix for the layer 1 of fast-nn

    x: data to compute dp matrix for
    r_bar: number of eigenvalues to keep
    '''
    p = np.shape(x)[1]
    covariance_matrix = x.T @ x
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)

    return dp_matrix

def instantiate_FAST_model(dataset,input_dim=config.INPUT_DIM, d_model=config.D_MODEL, nhead=config.NHEAD, num_layers=config.NUM_LAYERS, r_bar=config.R_BAR, width=config.WIDTH, fast_nn=True):
    print('loading data')
    
    train_size = int(len(dataset) * (1 - config.VALIDATION_SPLIT))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    if fast_nn: # checks if we're using fast-nn transformer or just base transformer
        dp_mat = compute_dp_mat(train_dataset.dataset.features)  # Compute pretrained dp matrix for Fast-NN
        model = FastNNTransformer(
            dp_mat=dp_mat,
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            r_bar=r_bar,
            width=width
        )

    else: # this probably doesn't work so maybe don't use it😛
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

    return model, train_loader, val_loader

def train_and_evaluate(dataset, model, train_loader, val_loader, fast_nn=True, learning_rate=config.LEARNING_RATE):
    try:
        # Load and preprocess data

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Initialize optimizer and scheduler (maybe unneccessary since we're using a subset)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # different from Fast-NN paper, works better w/ transformer probably
        
        # Training loop
        best_model = {'model': None, 
                      'best_mae': 999999999, 
                      'epoch': 0, 
                      'val_metrics': None}
        
        for epoch in range(config.NUM_EPOCHS):
            # Training 
            model.train()
            train_loss = 0
            for _, (X, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')):
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(X) # get model output
                loss = nn.MSELoss()(output, y) # compute L2 loss
                
                if fast_nn: # if using fast-nn get the L1 loss
                    reg_loss = model.regularization_loss(model, config.HP_TAU)
                    reg_loss = reg_loss / (config.BATCH_SIZE * config.SEQ_LEN)
                    loss += config.LAMBDA * reg_loss # scale by lambda
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP) # model's prone to exploding gradient so this is a quick solution
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Val 
            model.eval()
            val_metrics = evaluate(
                model, 
                val_loader, 
                feature_scaler=dataset.scaler,
                target_scaler=config.TARGET_SCALER,
                fast_nn=fast_nn
            )
            val_loss = val_metrics['loss']
            
            # print out metrics
            print(f"Epoch {epoch+1}:")
            print(f"Train Loss = {train_loss}")
            print(f"Val Loss = {val_loss}")
            print(f"Val MAE = {val_metrics['mae']}")
            print(f"Val RMSE = {val_metrics['rmse']}")
            
            # iterate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_metrics['mae'] < best_model['best_mae']:
                best_model['best_mae'] = val_metrics['mae']
                best_model['epoch'] = epoch
                best_model['val_metrics'] = val_metrics
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                print("Saved new best model!")
        
        print("Training completed!")
        print(f"Best model MAE: {best_model['best_mae']} at epoch {best_model['epoch']}")
        print(f"Best model metrics: {best_model['val_metrics']}")
        
    except Exception as e:
        print('training error 🙃')
        raise
