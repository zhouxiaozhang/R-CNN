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
        l2_loss=tf.constant(0.0)
        #embedding,ø…“‘≥¢ ‘”√one-hot
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
        for i, filter_size in enumerate(self._config.filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                #convolution layer1
                filter_shape=[filter_size,self._config.embedding_dim,1,self._config.num_filter1]
                W=tf.get_variable(name="W", shape=filter_shape,initializer=tf.truncated_normal_initializer(stddev = 0.1))
                #W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b=tf.get_variable(name="b",shape=[self._config.num_filter1],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
                conv=tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1,1,1,1],padding="VALID",name="conv1")
                #apply nonlinearity
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                #Maxpooling
                pooled=tf.nn.max_pool(h,ksize=[1,self._config.sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
                  # norm1
                # norm1 = tf.nn.lrn(pooled, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
                self.pooled_outputs.append(pooled)
                
        # Combine all the pooled features£¨1£¨1£¨1£¨1 -°∑1£¨1£¨1£¨3
        num_filters_total=self._config.num_filter1*len(self._config.filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
        # Final (unnormalized) scores and predictions    
        with tf.variable_scope("output"):
            W = tf.get_variable( "W",shape=[num_filters_total, self._config.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable(name="b",shape=[self._config.num_classes],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # CalculateMean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self._config.l2_reg_lambda * l2_loss
        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        
   