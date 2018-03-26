def get_config(opt):
    if opt == "test":
        return config_test()
    elif opt == "valid":
        return config_valid()
    else:
        return config()
    

class config(object):
    LEARNING_RATE_DECAY = 0.96
    max_epoch=4
    embedding_dim=100
    HIDDEN_SIZE = 500
    LAYERS_NUM = 2
    INIT_SCALE = 0.1
    CELL_FORGET_BIAS = 1.0
    MAX_GRAD = 1
    l2_reg_lamda=0.0
    batch_size=128
    num_epochs=100
    num_classes=1
    sequence_length=1000
    vocab_size=400
    learning_rate=0.001
    dropout_keep_prob=0.5
    
class config_valid(object):
    LEARNING_RATE_DECAY = 0.96
    EPOCH_NUM = 150
    max_epoch=4
    embedding_dim=100
    HIDDEN_SIZE = 500
    LAYERS_NUM = 2
    INIT_SCALE = 0.1
    CELL_FORGET_BIAS = 1.0
    MAX_GRAD = 1
    l2_reg_lamda=0.0
    batch_size=128
    num_epochs=100
    num_classes=1
    sequence_length=1000
    vocab_size=400
    learning_rate=1e-3
    dropout_keep_prob=1.0
    
class config_test(object):
    LEARNING_RATE_DECAY = 0.96
    EPOCH_NUM = 150
    max_epoch=4
    embedding_dim=100
    HIDDEN_SIZE = 500
    LAYERS_NUM = 2
    INIT_SCALE = 0.1
    CELL_FORGET_BIAS = 1.0
    MAX_GRAD = 1
    l2_reg_lamda=0.0
    batch_size=1
    num_epochs=100
    num_classes=1
    sequence_length=1000
    vocab_size=400
    learning_rate=1e-3
    dropout_keep_prob=1.0
    
