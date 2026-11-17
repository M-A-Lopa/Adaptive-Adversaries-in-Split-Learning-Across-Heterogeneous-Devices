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
    
    MODEL_NAME    = "KAGN" #change to "PyramidCNN"/"Vanilla_SL/KAGN" to switch models
    RUN_ATTACK    = True # change to True when you want to run the attacks otherwise False
    DEGREE        = 3 # Degree for KAGN model, only relevant if MODEL_NAME is "KAGN"

    

    #Unsplit attacks
    unsplit_main_iters=500 # Use 100
    unsplit_input_iters=100 # use 20 
    unsplit_model_iters=100 # use 20


        # ---- Multi-client split learning 
    # NUM_CLIENTS = 1 keeps the original single-client setup everywhere.
    NUM_CLIENTS      = 5        # 1 = single-client (original), 5 / 10 / ... = multi-client SplitFed
    DATA_SPLIT       = 'iid'    # 'iid' or 'noniid'
    NONIID_ALPHA     = 0.5      # Dirichlet concentration, only used when DATA_SPLIT = 'noniid'
    VICTIM_CLIENT    = 0        # client the server attacks (privacy attacks)
    MALICIOUS_CLIENT = 2        # client that plants the backdoor; a list such as [2, 7] for several
    POISON_START     = 3        # round from which the malicious client poisons
    SEED             = 0        # seed of the client split and of the training order