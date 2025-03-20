


class Hyperparams:
    '''Hyperparameters'''
    # data
    original_path_04 = '../data/pems04.npz'
    original_path_08 = '../data/pems08.npz'
    pkl_path_04  = '../data/pems04_120m.pkl'
    pkl_path_08 = '../data/pems08_120m.pkl'
    input_max_len = 288
    output_max_len = 24

    # training params
    batch_size = 32# alias = N
    lr = 0.001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory
    ckpt_path = './ckpt/weight'
    # model
    maxlen = input_max_len  # Maximum number of words in a sentence. alias = T.
    outputlen = output_max_len
    hidden_units = 128  # alias = C
    hd_ff = 4*hidden_units #inner cell number
    output_units = 1 #output units number
    num_blocks = 1  # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    if_validation =False
    valid_thresh = 100#if > then keep,else alter

    #radom mask
    mask_rate = 0.1
    if_rm = True