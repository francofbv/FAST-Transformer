# TimeSeries-Transformer

A PyTorch implementation of a Transformer model for time series prediction, combined with Fast-NN for feature selection.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data in the appropriate directory
2. Run the training script:
```bash
python scripts/train_optiver.py
```

## Project Structure

- `config/`: Configuration files
- `models/`: Model implementations
- `scripts/`: Training and evaluation scripts
- `utils/`: Utility functions and data loaders

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- SciPy >= 1.7.0
- tqdm >= 4.62.0 