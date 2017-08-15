#! /usr/bin/env python
# -*- coding: utf-8 -*-

def get_config(opt):
    if opt == "test":
        return config_test()
    elif opt == "valid":
        return config_valid()
    else:
        return config()
    

class config(object):
    embedding_dim=64
    filter_size1=3
    filter_sizes=[3,4,5]
    num_filter1=20
    num_filter2=40
    l2_reg_lamda=0.0
    batch_size=64
    num_epochs=100
    num_classes=2
    sequence_length=1000
    vocab_size=400
    one_hot=False
    learning_rate=1e-3
    dropout_keep_prob=0.5
    num_classes=2
    l2_reg_lambda=0.0
    
class config_valid(object):
    embedding_dim=64
    filter_size1=3
    filter_sizes=[3,4,5]
    num_filter1=20
    num_filter2=40
    l2_reg_lamda=0.0
    batch_size=64
    num_epochs=100
    num_classes=2
    sequence_length=1000
    vocab_size=400
    one_hot=False
    learning_rate=1e-3
    dropout_keep_prob=1.0
    num_classes=2
    l2_reg_lambda=0.0
    
class config_test(object):
    embedding_dim=64
    filter_size1=3
    filter_sizes=[3,4,5]
    num_filter1=20
    num_filter2=40
    l2_reg_lamda=0.0
    batch_size=1
    num_epochs=100
    num_classes=2
    sequence_length=1000
    vocab_size=400
    one_hot=False
    learning_rate=1e-3
    dropout_keep_prob=1.0
    num_classes=2
    l2_reg_lambda=0.0
    
