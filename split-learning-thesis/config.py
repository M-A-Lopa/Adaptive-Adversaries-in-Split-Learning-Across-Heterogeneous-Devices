# Central configuration for all hyperparameters and settings
# Change values here only — all other files read from this file

class Config:

    # ----------------------Dataset--------------------------
    DATASET       = 'CIFAR10'   # Switch to 'MNIST' anytime
    DATA_DIR      = './data'
    NUM_CLASSES   = 10

    # ----------------------Training--------------------------
    BATCH_SIZE    = 64
    EPOCHS        = 50
    LEARNING_RATE = 0.001

    # ----------------------Split Learning--------------------------
    # CUT_LAYER = 2 means client runs Conv1 + Conv2
    # Server runs Conv3 + FC layers
    # You will change this value during heterogeneous setup
    CUT_LAYER     = 2

    # ----------------------Device--------------------------
    DEVICE        = 'cuda'      # Automatically falls back to cpu if cuda unavailable

    # ----------------------Save Paths--------------------------
    SAVE_DIR      = './checkpoints'
    RESULTS_DIR   = './results'
