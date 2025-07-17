# Model configuration
'''
model configuration to allow for default values to be set and convenient hyperparameter tuning
'''
class Config:
    '''
    Configuration for the model

    Model parameters:
    INPUT_DIM: input dimension
    SEQ_LEN: sequence length
    D_MODEL: model dimension
    NHEAD: number of attention heads
    NUM_LAYERS: number of transformer layers
    BATCH_SIZE: batch size for training
    NUM_EPOCHS: number of training epochs
    LEARNING_RATE: learning rate for optimizer
    HP_TAU: tau value for regularization loss
    R_BAR: number of eigenvalues to keep
    WIDTH: width of the fast-nn model
    LAMBDA: optimal lambda for fast-nn (by hyperparameter search)
    PENALIZE_WEIGHTS: whether to penalize weights for fast-nn (L1 & L2)
    LAGS: lags for dataset composition

    Training parameters:
    VALIDATION_SPLIT: validation split ratio
    GRADIENT_CLIP: gradient clipping value
    '''
    # Model params 
    INPUT_DIM = 7  # ETTh1 has 7 features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
    SEQ_LEN = 96   # Standard ETT sequence length
    D_MODEL = 64  # Reduced for multivariate efficiency
    NHEAD = 8
    NUM_LAYERS = 3  # Reduced to prevent overfitting
    BATCH_SIZE = 64  # Increased for better gradient estimates
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5  # Reduced for stable multivariate training
    HP_TAU = 0.1
    R_BAR = 3      # Optimal for 7-dimensional input (keep ~40% of components)
    WIDTH = 32     # Significantly reduced for multivariate efficiency
    LAMBDA = 0.5   # Reduced from 4 to prevent over-regularization
    PENALIZE_WEIGHTS = False
    LAGS = [5, 10, 20, 50]
    
    # ETTh1 specific parameters
    PRED_LENS = [96]  # Testing with just one horizon
    MULTIVARIATE = True   # Fixed: Set to True for multivariate forecasting
    OUTPUT_DIM = 7  # Fixed: 7 for multivariate (all features)
    
    # Training params 
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP = 1.0
    
    # Data paths
    DATA_PATH = 'data/ETTh1.csv'
    
config = Config()
