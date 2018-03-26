#! /usr/bin/env python
# -*- coding: utf-8 -*-
       
import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import logging
import pickle
import CNN
import math
from CNN import malware_RNN
from config import get_config
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import accuracy_score

#Data parameters
tf.flags.DEFINE_string("data_path", "/home/zx/gram3_gain1000/data_process_data_original_value_train","data_path")
tf.flags.DEFINE_string("save_path" , "/home/zx/RCNN/CNN/session_save/","Model output directory.")
tf.flags.DEFINE_string("board_path", "/home/zx/RCNN/CNN/tensor_board/","Tensor board output directory.")
tf.flags.DEFINE_string("log_path", "/home/zx/RCNN/CNN/log.log", "log output path.")
#Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement",True,"Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement",False,"Log placement of ops on devices")
#train Parameters
tf.flags.DEFINE_string("evaluate_every",2,"Evaluate model on dev set after this many epochs")

FLAGS =tf.flags.FLAGS

#log
LOG = None
def init_logger():
    global LOG

    LOG = logging.getLogger('seq')
    LOG.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filename=FLAGS.log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)

def load_data(file_name):
    with open(file_name, "rb") as f:
        raw_x, raw_y = pickle.load(f)
        l = len(raw_x)
        train_data = []
        train_lable=[]
        valid_data = []
        valid_lable=[]
        pos_1 = int(l * 0.9)
        train_data = raw_x[:pos_1]
        train_lable=raw_y[:pos_1]
        valid_data = raw_x[pos_1: ]
        valid_lable = raw_y[pos_1: ]
        return  train_data,valid_data,train_lable,valid_lable
    
def batch_iter(x_data,y_data, batch_size, num_epochs, shuffle=True):
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    data_size =len(x_data)
    num_batches_per_epoch = int((data_size-1) / batch_size)+1
    #for epoch in range(num_epochs):
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
        
def real_len(batches):
    real=[]
    for batch in batches:
        if min(batch)>=1:
            result=1000
        else:
            result=np.argmin(batch)
        real.append(result) 
    return real
                    
def run_epoch(
        session,
        x_batch,
        y_batch,
        dropout,
        model,
        global_step,
        summary_op,
        eval_op=None,
        verbose=False
       
):
    #feed,give
    feed_dict={
        model.input_x:x_batch,
        model.input_y:y_batch,
        model.dropout_keep_prob:dropout,
        model.batch_size:len(x_batch),
        model.real_len:real_len(x_batch)
              }
    #out
    fetches = {
        "loss": model.loss,
        "global_step":global_step,
        "summary_op" :summary_op,
        "y_pre"      :model.y_pre
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    
    fetches_ret = session.run(fetches, feed_dict)
    loss = fetches_ret["loss"]
    y_p=fetches_ret["y_pre"]
    auc=roc_auc_score(y_batch, y_p)
    for i,y_acc in enumerate(y_p):
            if y_acc>0.5:
                y_p[i]=1
            else:
                y_p[i]=0
    acc=accuracy_score(y_batch, y_p)
    if verbose:
        LOG.info( "step: %d,loss: %.3f auc: %.3f acc:%.3f" 
                  % ( fetches_ret["global_step"], fetches_ret["loss"],auc,acc)
                 )
    return auc, fetches_ret["loss"],fetches_ret["summary_op"]

def train():
    
    #training 
    x_train,x_dev,y_train,y_dev=load_data(FLAGS.data_path)
    init_logger()
    train_config=get_config("train")
    valid_config=get_config("valid")
    with tf.Graph().as_default():
        session_conf=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
        sess=tf.Session(config=session_conf)
        with sess.as_default():
            cnn=malware_RNN(config=train_config)
            global_step = tf.contrib.framework.get_or_create_global_step()
            trainable_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients( cnn.loss, trainable_vars),train_config.MAX_GRAD)
            optimizer = tf.train.AdamOptimizer(train_config.learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, trainable_vars),global_step=tf.contrib.framework.get_or_create_global_step())
            #optimizer=tf.train.RMSPropOptimizer(train_config.learning_rate,decay=0.9)
            #grads_and_vars=optimizer.compute_gradients(cnn.loss)
            #train_op=optimizer.apply_gradients(grads_and_vars,global_step)
            #keep track of gradient values and sparsity
            grad_summaries=[]
            for g,v in zip(grads, trainable_vars):
                if g is not None:
                    grad_hist_summary=tf.summary.histogram("{}/grad/hist".format(v.name),g)
                    sparsity_summary=tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged=tf.summary.merge(grad_summaries)
            #Summaries for loss and acc
            loss_summary=tf.summary.scalar("loss",cnn.loss)
            
            #Train summaries
            train_summary_op=tf.summary.merge([grad_summaries_merged,loss_summary])
            train_summary_dir=os.path.join(FLAGS.board_path,"summaries","train")
            train_summary_write=tf.summary.FileWriter(train_summary_dir,sess.graph)
            #Dev summaries
            dev_summary_op=loss_summary
            dev_summary_dir=os.path.join(FLAGS.board_path,"summaries","dev")
            dev_summary_write=tf.summary.FileWriter(dev_summary_dir,sess.graph)
            
           
            #init
            checkpoint_dir=os.path.join(FLAGS.save_path,"checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix=os.path.join(checkpoint_dir,"model")
            
            saver=tf.train.Saver(tf.all_variables())
            sess.run(tf.global_variables_initializer())
            #generate batches
            
            best_acc,best_at_step=0,0
            #training loop
            accuracy_total=0.0
            loss_total=0.0
            for epoch_id in range(0, train_config.num_epochs):
                #accuracy_total=accuracy_total/ (int((len(x_train)-1) / train_config.batch_size)+1)
                #loss_total=loss_total/(int((len(x_train)-1) / train_config.batch_size)+1)
                #LOG.info("\ntrain_epoch:")
                #LOG.info( "loss_total: %.3f accuracy_total: %.3f" 
                  #% ( loss_total,accuracy_total)
                #)
                accuracy_total=0.0
                loss_total=0.0
                batches=batch_iter(x_train,y_train,train_config.batch_size,train_config.num_epochs)
                for batch_x,batch_y in batches:
                    acc_train,loss_train,summaries=run_epoch( sess,batch_x,batch_y,dropout=train_config.dropout_keep_prob,model=cnn, eval_op=train_op,verbose=True, global_step= global_step,summary_op=train_summary_op)
                    accuracy_total+=acc_train
                    loss_total+=loss_train
                    current_step=tf.train.global_step(sess,global_step)
                    train_summary_write.add_summary(summaries,epoch_id)
                accuracy_total=accuracy_total/ (int((len(x_train)-1) / train_config.batch_size)+1)
                loss_total=loss_total/(int((len(x_train)-1) / train_config.batch_size)+1)
                LOG.info("\ntrain_epoch:%.3f" %(epoch_id))
                LOG.info( "loss_total: %.3f accuracy_total: %.3f" % ( loss_total,accuracy_total))
                if epoch_id%FLAGS.evaluate_every==0:
                    LOG.info("\nEvaluation: %.3f" %(epoch_id))
                    acc,loss,summaries=run_epoch( sess,x_dev,y_dev,dropout=valid_config.dropout_keep_prob,model=cnn,verbose=True, global_step= global_step,summary_op=dev_summary_op) 
                    dev_summary_write.add_summary(summaries,epoch_id)
                    if acc>=best_acc:
                        best_acc,best_at_step=acc,epoch_id
                        path=saver.save(sess,checkpoint_prefix,global_step=epoch_id)
                        LOG.info("Saving model to %s at epoch %d." % (path,epoch_id))

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data file")
    train()

if __name__ == '__main__':
    tf.app.run()