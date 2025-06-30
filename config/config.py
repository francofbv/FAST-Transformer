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
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    BATCH_SIZE = 32  # Reduced for longer sequences
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4  # Slightly higher for ETT dataset
    HP_TAU = 0.1
    R_BAR = 6      # Reduced proportionally for 7 features (was 12 for 28 features)
    WIDTH = 128    # Reduced for smaller input dimension
    LAMBDA = 4
    PENALIZE_WEIGHTS = True
    LAGS = [5, 10, 20, 50]
    
    # ETTh1 specific parameters
    PRED_LENS = [96, 192, 336, 720]  # Standard ETT forecasting horizons
    
    # Training params 
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP = 1.0
    
config = Config()
