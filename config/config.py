# Model configuration
class Config:
    # Model parameters
    SEQ_LEN = 60  # Sequence length for the transformer model
    D_MODEL = 64  # Dimension of the model
    NHEAD = 4     # Number of attention heads
    NUM_LAYERS = 2  # Number of transformer layers
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    HP_TAU = 1e-1
    R_BAR = 5
    WIDTH = 32
    CHOICE_LAMBDA = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    PENALIZE_WEIGHTS = False 

config = Config()
