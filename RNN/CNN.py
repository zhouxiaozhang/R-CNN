#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np

class malware_RNN(object):
    def __init__(self,config):
        self._config=config
        self._data_type = tf.float32
        self.input_x=tf.placeholder(tf.int32,[None,self._config.sequence_length],name="input_x")    
        self.input_y=tf.placeholder(tf.float32,[None,self._config.num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')
        self.batch_size = tf.placeholder(tf.int32, [],name="batch_size")
        l2_loss=tf.constant(0.0)
        #embedding,ø…“‘≥¢ ‘”√one-hot
        with tf.device("/cpu:0"),tf.name_scope("embedding"):
            embedding = tf.get_variable(name="embedding",shape=[self._config.vocab_size,self._config.embedding_dim],initializer=tf.random_uniform_initializer(-1.0,1.0),dtype=self._data_type)
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        with tf.variable_scope("LSTM"): 
            inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
            cell = tf.nn.rnn_cell.LSTMCell(self._config.HIDDEN_SIZE,forget_bias=self._config.CELL_FORGET_BIAS,state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            cells = tf.nn.rnn_cell.MultiRNNCell([cell] * self._config.LAYERS_NUM,state_is_tuple=True)
            self._initial_state = cells.zero_state( self.batch_size,self._data_type)
            outputs, states = tf.nn.dynamic_rnn(cells,inputs,initial_state=self._initial_state,parallel_iterations=1,dtype=tf.float32,
            sequence_length=self.real_len,time_major=False)
        with tf.variable_scope("max-pooling"): 
            #masking the output vector,batch,hidden
            index = tf.range(0, self.batch_size) * self._config.sequence_length + (self.real_len - 1)
            flat = tf.reshape(outputs, [-1, self._config.HIDDEN_SIZE])
            last=tf.gather(flat, index)
            
            #max_pooling batch_size,2*hidden_size
            trans=tf.transpose(outputs, perm=[0, 2, 1])
            max_pool=tf.reduce_max(trans,axis=2)
            last_output=last
            max_pooling = tf.concat(1, [max_pool, last_output])
        with tf.variable_scope("classifier"): 
            #init
            init_w=tf.truncated_normal_initializer(stddev = 0.1)
            init_b=tf.constant_initializer(value=0.1, dtype=tf.float32)
            softmax_w = tf.get_variable("softmax_w",[self._config.HIDDEN_SIZE*2, 1], dtype=self._data_type,initializer=init_w )
            softmax_b = tf.get_variable( "softmax_b",[1], dtype=self._data_type,initializer=init_b)
            l2_loss += tf.nn.l2_loss(softmax_w)
            l2_loss += tf.nn.l2_loss(softmax_b)
            l2_loss += tf.nn.l2_loss(embedding)
            log = tf.matmul(max_pooling, softmax_w) + softmax_b 
        with tf.variable_scope("loss"):    
            cross_entropy =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(log, self.input_y))
            self.y_pre=tf.sigmoid(log,name="y_pred") 
            self.loss=cross_entropy+self._config.l2_reg_lamda*l2_loss
            self.final_state = states         
    def batch_iter(x_data,y_data, batch_size, num_epochs, shuffle=True):
        x_data=np.array(x_data)
        y_data=np.array(y_data)
        data_size =len(x_data)
        num_batches_per_epoch = int(len(x_data)-1/batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices=np.random.permutation(data_size)
                shuffled_data_x=x_data[shuffle_indices]
                shuffled_data_y=y_data[shuffle_indices]
                
            else:
                shuffled_data_x=x_data
                shuffled_data_y=y_data
            for batch_num in range(num_batches_per_epoch):
                start_index=batch_num*batch_size
                end_index=min((batch_num+1)*batch_size,data_size)
                yield shuffled_data_x[start_index:end_index],shuffled_data_y[start_index:end_index]
                
        
       

        

        
                
                
            
            
            
        
                
            
            
            
        
    
       