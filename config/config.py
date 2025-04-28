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
    INPUT_DIM = 28
    SEQ_LEN = 20
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-5
    HP_TAU = 0.1
    R_BAR = 12
    WIDTH = 256
    LAMBDA = 4
    PENALIZE_WEIGHTS = True
    LAGS = [5, 10, 20, 50]
    
    # Training params 
    VALIDATION_SPLIT = 0.2
    GRADIENT_CLIP = 1.0
    
config = Config()
