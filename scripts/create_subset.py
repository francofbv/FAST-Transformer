import pandas as pd
import numpy as np
import os

def create_subset(input_path, output_path, num_time_points=50, num_stocks=10):
    """
    Create a subset of the training data with specified number of time points and stocks
    
    Args:
        input_path: Path to the original training data
        output_path: Path to save the subset
        num_time_points: Number of time points to include
        num_stocks: Number of stocks to include
    """
    # Read the original data
    df = pd.read_csv(input_path)
    
    # Get unique time_ids and stock_ids
    unique_times = sorted(df['time_id'].unique())
    unique_stocks = sorted(df['stock_id'].unique())
    
    # Select the first num_time_points time points
    selected_times = unique_times[:num_time_points]
    
    # Select the first num_stocks stocks
    selected_stocks = unique_stocks[:num_stocks]
    
    # Filter the data
    subset_df = df[
        (df['time_id'].isin(selected_times)) & 
        (df['stock_id'].isin(selected_stocks))
    ]
    
    # Sort by time_id and stock_id
    subset_df = subset_df.sort_values(['time_id', 'stock_id'])
    
    # Save the subset
    subset_df.to_csv(output_path, index=False)
    
    # Print information about the subset
    print(f"Created subset with:")
    print(f"- {len(subset_df)} rows")
    print(f"- {len(subset_df['time_id'].unique())} unique time points")
    print(f"- {len(subset_df['stock_id'].unique())} unique stocks")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Paths
    input_path = "/Users/francovidal/Desktop/School_Work/spring_25/AI2/AI2-Final-Project/train.csv"
    output_path = "data/train_subset.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create subset with 50 time points and 10 stocks
    create_subset(input_path, output_path, num_time_points=50, num_stocks=10) 