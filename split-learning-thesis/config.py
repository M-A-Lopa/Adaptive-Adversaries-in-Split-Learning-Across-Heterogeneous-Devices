class Config:

    DATASET       = 'MNIST'   # Switch to 'MNIST'/'CIFAR10' anytime 
    DATA_DIR      = './data'
    NUM_CLASSES   = 10

    BATCH_SIZE    = 64
    EPOCHS        = 50
    LEARNING_RATE = 0.001

    CUT_LAYER     = 4

    DEVICE        = 'cuda' 

    SAVE_DIR      = './checkpoints'
    RESULTS_DIR   = './results'
    
    MODEL_NAME    = "PyramidCNN" #change to "PyramidCNN"/"Vanilla_SL/KAGN" to switch models
    RUN_ATTACK    = True # change to True when you want to run the attacks otherwise False
    DEGREE        = 3 # Degree for KAGN model, only relevant if MODEL_NAME is "KAGN"
