#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np

class malware_CNN(object):
    def __init__(self,config):
        self._config=config
        self.input_x=tf.placeholder(tf.int32,[None,self._config.sequence_length],name="input_x")    
        self.input_y=tf.placeholder(tf.float32,[None,self._config.num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.real_len=tf.placeholder(tf.int32,[None],name="real_len")
        self.batch_size=tf.placeholder(tf.int32,[])
        self.pad=tf.placeholder(tf.float32,[None,1,self._config.embedding_dim,1],name="pad")
        
        l2_loss=tf.constant(0.0)
        #embedding,可以尝试用one-hot,char_expanded [batch,sequence_length,embeding_dim,1]
        with tf.device("/cpu:0"),tf.name_scope("embedding"):
            if not self._config.one_hot :
                self.embedded=tf.get_variable(name="embedded",shape=[self._config.vocab_size,self._config.embedding_dim],initializer=tf.random_uniform_initializer(-1.0,1.0))
                #self.embedded=tf.Variable(tf.random_uniform([self._config.vocab_size,self._config.embedding_dim],-1.0,1.0),name="embedded")
                self.embedded_chars=tf.nn.embedding_lookup(self.embedded,self.input_x)
                self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)
            else:
                self.embedded_chars=tf.contrib.keras.backend.one_hot(self.input_x,vocab_size)
                self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)
        # with  tf.variable_scope('conv1') as scope:
        self.pooled_outputs = []
        reduced=np.int32(np.ceil(self._config.sequence_length*1.0/self._config.max_pool_size))
        for i, filter_size in enumerate(self._config.filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                #zero padding in dimension sequence_length
                num_prio=(filter_size-1)//2
                num_post=(filter_size-1)-num_prio
                pad_prio=tf.concat([self.pad]*num_prio,1)
                pad_post=tf.concat([self.pad]*num_post,1)
                emd_pad=tf.concat([pad_prio,self.embedded_chars_expanded,pad_post],1)
                #convolution layer1
                filter_shape=[filter_size,self._config.embedding_dim,1,self._config.num_filter1]
                W=tf.get_variable(name="W", shape=filter_shape,initializer=tf.truncated_normal_initializer(stddev = 0.1))
                #W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b=tf.get_variable(name="b",shape=[self._config.num_filter1],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
                conv=tf.nn.conv2d(emd_pad,W,strides=[1,1,1,1],padding="VALID",name="conv1")
                print("conv",conv.get_shape())
                #apply nonlinearity
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                #Maxpooling
                pooled=tf.nn.max_pool(h,ksize=[1,self._config.max_pool_size,1,1],strides=[1,self._config.max_pool_size,1,1],padding="SAME",name="pool")
                print("pooled_max",pooled.get_shape())
  
                # norm1
                # norm1 = tf.nn.lrn(pooled, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
                pooled=tf.reshape(pooled,[-1,reduced,self._config.num_filter1])
                print("pooled",pooled.get_shape())
                self.pooled_outputs.append(pooled)
                
        # Combine all the pooled features，1，1，1 -》1，1，3,final [batch reduced（相当于sequence） num_filter1*3]
        #num_filters_total=self._config.num_filter1*len(self._config.filter_sizes)
        #（batch reduced（相当于sequence） num_filter1*3-》concat）
        self.h_pool = tf.concat(self.pooled_outputs, 2)
        print("self.h_pool",self.h_pool.get_shape())

        #self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        # Add dropout
        with tf.variable_scope("dropout"):
            inputs = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)
            # self.h_drop = tf.nn.dropout(self.h_pool,self.dropout_keep_prob)
        with tf.variable_scope("LSTM"):
            rnn_cell1 = tf.nn.rnn_cell.LSTMCell(self._config.HIDDEN_SIZE,forget_bias=self._config.CELL_FORGET_BIAS,state_is_tuple=True)
            rnn_cell2 = tf.nn.rnn_cell.LSTMCell(self._config.HIDDEN_SIZE,forget_bias=self._config.CELL_FORGET_BIAS,state_is_tuple=True)
            rnn_cell1= tf.nn.rnn_cell.DropoutWrapper(rnn_cell1, output_keep_prob=self.dropout_keep_prob)
            stack_rnn = [rnn_cell1]
            stack_rnn.append(rnn_cell2)
            cells = tf.nn.rnn_cell.MultiRNNCell(stack_rnn,state_is_tuple=True )
            #cell = tf.nn.rnn_cell.LSTMCell(self._config.HIDDEN_SIZE,forget_bias=self._config.CELL_FORGET_BIAS,state_is_tuple=True)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            #cells = tf.nn.rnn_cell.MultiRNNCell([cell] * self._config.LAYERS_NUM,state_is_tuple=True )
            self._initial_state = cells.zero_state(self.batch_size,tf.float32)
            outputs, states = tf.nn.dynamic_rnn(
            cells,
            inputs,
            initial_state=self._initial_state,
            parallel_iterations=1,
            dtype=tf.float32,
            sequence_length=self.real_len,
            time_major=False )
        with tf.variable_scope("Max-pooling"):
            #masking the output vector,batch,hidden
            index = tf.range(0, self.batch_size) * reduced + (self.real_len - 1)
            flat = tf.reshape(outputs, [-1, self._config.HIDDEN_SIZE])
            last=tf.gather(flat, index)

            #max_pooling batch_size,2*hidden_size
            trans=tf.transpose(outputs, perm=[0, 2, 1])
            max_pool=tf.reduce_max(trans,axis=2)
            last_output=last
            max_pooling = tf.concat([max_pool, last_output],1)
            
        # Final (unnormalized) scores and predictions    
        with tf.variable_scope("output"):
            #init
            softmax_w = tf.get_variable("softmax_w",
            shape=[self._config.HIDDEN_SIZE*2, self._config.num_classes],initializer=tf.truncated_normal_initializer(stddev = 0.1))
            softmax_b = tf.get_variable( "softmax_b",
            shape=[self._config.num_classes],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
            # W = tf.get_variable( "W",shape=[num_filters_total, self._config.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            #b=tf.get_variable(name="b",shape=[self._config.num_classes],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
            l2_loss += tf.nn.l2_loss(softmax_w)
            l2_loss += tf.nn.l2_loss(softmax_b)
            self.scores = tf.nn.xw_plus_b(max_pooling, softmax_w,  softmax_b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # CalculateMean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self._config.l2_reg_lambda * l2_loss
        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        
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
                
        
   
