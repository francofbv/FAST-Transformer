import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.sparse.linalg import eigsh

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.config import config
from models.fastnn_transformer import FastNNTransformer
from utils.dataloader import ETTh1Dataset
from scripts.evaluate import evaluate_etth1

def compute_dp_mat(data, r_bar=config.R_BAR):
    '''Compute diversified projection matrix for FAST-NN'''
    p = data.shape[1]
    covariance_matrix = data.T @ data
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)
    return dp_matrix

def test_data_loading():
    '''Test that data loading works correctly'''
    print("Testing data loading...")
    
    # Use the real dataset path
    data_path = '/Users/francovidal/Desktop/personal_projects/data/ETTh1.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.exists(data_path):
        print(f"Creating dummy ETTh1 data at {data_path}")
        # Create dummy ETTh1 data for testing
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Generate 2000 hourly timestamps (about 2.5 months of data)
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(2000)]
        
        # Generate dummy features (7 features for ETTh1)
        np.random.seed(42)  # For reproducible dummy data
        n_samples = len(dates)
        
        data = {
            'date': dates,
            'HUFL': np.random.randn(n_samples) * 0.5 + 10,  # High UseFul Load
            'HULL': np.random.randn(n_samples) * 0.3 + 8,   # High UseLess Load  
            'MUFL': np.random.randn(n_samples) * 0.4 + 12,  # Middle UseFul Load
            'MULL': np.random.randn(n_samples) * 0.2 + 9,   # Middle UseLess Load
            'LUFL': np.random.randn(n_samples) * 0.3 + 7,   # Low UseFul Load
            'LULL': np.random.randn(n_samples) * 0.1 + 5,   # Low UseLess Load
            'OT': np.random.randn(n_samples) * 0.6 + 15,    # Oil Temperature (target)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        print(f"Dummy data created with {len(df)} samples and {len(df.columns)-1} features")
    
    # Test dataset creation
    try:
        train_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=96, split='train')
        val_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=96, split='val') 
        test_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=96, split='test')
        
        # Set scalers
        val_dataset.set_scaler(train_dataset.scaler)
        test_dataset.set_scaler(train_dataset.scaler)
        
        print(f"✅ Data loading successful!")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Test a single batch
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        X, y = next(iter(train_loader))
        print(f"Sample batch shapes - X: {X.shape}, y: {y.shape}")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None, None

def test_model_creation(train_dataset):
    '''Test model creation and forward pass'''
    print("\nTesting model creation...")
    
    try:
        # Get some training data for dp_mat computation
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        train_features = []
        
        # Collect a few batches for dp_mat computation
        for i, (X, _) in enumerate(train_loader):
            if i >= 3:  # Just use a few batches for testing
                break
            batch_size, seq_len, num_features = X.shape
            train_features.append(X.reshape(-1, num_features).numpy())
        
        train_features = np.vstack(train_features)
        print(f"Collected {train_features.shape[0]} samples with {train_features.shape[1]} features")
        
        # Compute dp_mat
        dp_mat = compute_dp_mat(train_features)
        print(f"DP matrix shape: {dp_mat.shape}")
        
        # Create model
        model = FastNNTransformer(
            dp_mat=dp_mat,
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            r_bar=config.R_BAR,
            width=config.WIDTH,
            pred_len=96
        )
        
        print(f"✅ Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        X_test, y_test = next(iter(train_loader))
        X_test, y_test = X_test.to(device), y_test.to(device)
        
        with torch.no_grad():
            output = model(X_test)
            print(f"Forward pass successful! Input: {X_test.shape}, Output: {output.shape}, Target: {y_test.shape}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_training_loop(model, train_dataset, val_dataset, device):
    '''Test a few training iterations'''
    print("\nTesting training loop...")
    
    try:
        # Create small data loaders for quick testing
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        model.train()
        
        # Test a few training steps
        print("Running 3 training batches...")
        for i, (X, y) in enumerate(train_loader):
            if i >= 3:  # Just test 3 batches
                break
                
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(X, is_training=True)
            loss = criterion(output, y)
            
            # Add regularization
            reg_loss = model.regularization_loss(model, config.HP_TAU)
            total_loss = loss + config.LAMBDA * reg_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            print(f"  Batch {i+1}: Loss = {loss.item():.6f}, Reg Loss = {reg_loss.item():.6f}, Total = {total_loss.item():.6f}")
        
        # Test evaluation
        print("Testing evaluation...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                if i >= 2:  # Just test 2 batches
                    break
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(f"Average validation loss: {avg_val_loss:.6f}")
        
        print("✅ Training loop test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_function(model, test_dataset, device):
    '''Test the evaluation function'''
    print("\nTesting evaluation function...")
    
    try:
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Limit to a few batches for quick testing
        limited_batches = []
        for i, batch in enumerate(test_loader):
            if i >= 3:  # Just test with 3 batches
                break
            limited_batches.append(batch)
        
        # Create a temporary loader with limited data
        class TempLoader:
            def __init__(self, batches):
                self.batches = batches
            def __iter__(self):
                return iter(self.batches)
            def __len__(self):
                return len(self.batches)
        
        temp_loader = TempLoader(limited_batches)
        
        # Test evaluation
        metrics = evaluate_etth1(model, temp_loader, device)
        
        print("✅ Evaluation function test successful!")
        print(f"Sample metrics: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    '''Run all tests'''
    print("🧪 FAST-Transformer ETTh1 Quick Test")
    print("=" * 50)
    
    # Test 1: Data loading
    train_dataset, val_dataset, test_dataset = test_data_loading()
    if not train_dataset:
        print("❌ Cannot proceed without working data loading")
        return
    
    # Test 2: Model creation
    model, device = test_model_creation(train_dataset)
    if not model:
        print("❌ Cannot proceed without working model")
        return
    
    # Test 3: Training loop
    training_ok = test_training_loop(model, train_dataset, val_dataset, device)
    if not training_ok:
        print("❌ Training loop has issues")
        return
    
    # Test 4: Evaluation function
    eval_ok = test_evaluation_function(model, test_dataset, device)
    if not eval_ok:
        print("❌ Evaluation function has issues")
        return
    
    print("\n🎉 All tests passed! The setup is ready for full training.")
    print("\nNext steps:")
    print("1. Add real ETTh1.csv data to data/ directory")
    print("2. Run: python scripts/train_etth1.py")
    print("3. Run: python scripts/benchmark_etth1.py")

if __name__ == "__main__":
    main()