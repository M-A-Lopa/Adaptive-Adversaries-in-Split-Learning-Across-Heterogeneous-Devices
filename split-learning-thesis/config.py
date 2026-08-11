class Config:

    DATASET       = 'CIFAR10'   # Switch to 'MNIST'/'CIFAR10' anytime 
    DATA_DIR      = './data'
    NUM_CLASSES   = 10

    BATCH_SIZE    = 64
    EPOCHS        = 50
    LEARNING_RATE = 0.001

    CUT_LAYER     = 2

    DEVICE        = 'cuda' 

    SAVE_DIR      = './checkpoints'
    RESULTS_DIR   = './results'
