# Model configuration
class Config:
    '''
    Configuration for the model
    '''
    # Model parameters
    INPUT_DIM = 28  # Increased to handle more features
    SEQ_LEN = 20  # Reduced sequence length to match dataset size
    D_MODEL = 256  # Increased model dimension for more complex patterns
    NHEAD = 8      # Number of attention heads
    NUM_LAYERS = 6  # Increased number of transformer layers
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    HP_TAU = 1
    R_BAR = 15
    WIDTH = 256    # Increased width for Fast-NN
    CHOICE_LAMBDA = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    PENALIZE_WEIGHTS = True
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    GRADIENT_CLIP = 1.0
    
    # Data parameters
    TRAIN_DATA_PATH = 'data/train.csv'
    TEST_DATA_PATH = 'data/test.csv'
    SUBMISSION_PATH = 'submissions'

config = Config()
