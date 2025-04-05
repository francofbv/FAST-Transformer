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

config = Config()
