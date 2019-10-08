#!/usr/bin/env python
# coding: utf-8

# # 1. Import library

# In[1]:


import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt


# # 2. Declare function

# In[2]:


# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)


def conv_layer_02(x, filter_size=5, num_filters=6, stride=1, name='conv',ismaxpool=True):
    """
    Create a 2D convolution layer
    :param x: input from previous layer
    :param filter_size: size of each filter
    :param num_filters: number of filters (or output feature maps)
    :param stride: filter stride
    :param name: layer name
    :param ismaxpool: ismaxpool= True
    :return: The output array
    """
    with tf.variable_scope(name):
        #prepare
        num_in_channel = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        # computing
        W = weight_variable(shape=shape)
        b = bias_variable(shape=[num_filters])
        conv = tf.nn.conv2d(x, W,
                             strides=[1, stride, stride, 1],
                             padding="SAME")+b
        act=tf.nn.relu(conv)
        
        # summary
        tf.summary.histogram('weight', W)
        tf.summary.histogram('bias', b)
        tf.summary.histogram('activation',act)
        # return
        if ismaxpool:
            return tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        else:
            return act
#
def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat
#
def fc_layer_02(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        # prepare
        in_dim = x.get_shape()[1]
        # Computing
        W = weight_variable(shape=[in_dim, num_units])
        b = bias_variable(shape=[num_units])
        act = tf.matmul(x, W)+b
        # summary
        tf.summary.histogram('weight', W)
        tf.summary.histogram('bias', b)
        tf.summary.histogram('activation',act)
        # return
        if use_relu:
            act = tf.nn.relu(act)
        return act
#
def imshows(images,labels,map=[6,6],name='figure_name'):
    map=np.asarray(map)
    p=np.product(map)
    plt.figure(figsize=list(map*2))
    for i in range(p):
        plt.subplot(map[0],map[1],i+1)
        plt.imshow(images[i].squeeze())
        plt.title(str(labels[i]))
        plt.tick_params(labelbottom=False, labelleft=False)
    plt.suptitle(name)
    plt.show()


# # 3. Load mnist dataset

# In[3]:


# loading mnist dataset
mnist=tf.contrib.learn.datasets.mnist.read_data_sets(train_dir='data',reshape=False,one_hot=True)
# train
X_train=mnist.train.images
y_train=mnist.train.labels
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
# validation
X_valid=mnist.validation.images
y_valid=mnist.validation.labels
print(f'X_valid shape: {X_valid.shape}')
print(f'y_valid shape: {y_valid.shape}')
# test
X_test=mnist.test.images
y_test=mnist.test.labels
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')


# # 3.1 showing the mnist dataset

# In[4]:


flagshow=True
if flagshow:
    imshows(X_train,np.argmax(a=y_train,axis=1),map=[6,6],name='Train')
    imshows(X_valid,np.argmax(a=y_valid,axis=1),map=[6,6],name='valid')
    imshows(X_test,np.argmax(a=y_test,axis=1),map=[6,6],name='test')
'''
X_train shape: (55000, 28, 28, 1)
y_train shape: (55000, 10)
X_valid shape: (5000, 28, 28, 1)
y_valid shape: (5000, 10)
X_test shape: (10000, 28, 28, 1)
y_test shape: (10000, 10)
'''


# # 4. define graph and training model

# In[5]:


# config parameter
lr=0.001
logdir='.\\logdir'
epochs=10
batch_size=100
freq01=10
freq02=500
global_step=0
stopping_accuracy=0.98
def model():
    global global_step
    #
    tf.reset_default_graph()
    sess=tf.Session()
    #
    x=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1],name='x')
    tf.summary.image(name='x',tensor=x,max_outputs=3,collections=None,family=None)
    y=tf.placeholder(dtype=tf.float32,shape=[None,10],name='y')
    conv1=conv_layer_02(x,filter_size=5,num_filters=6,stride=1,name='conv1',ismaxpool=True)
    conv2=conv_layer_02(conv1,filter_size=5,num_filters=16,stride=1,name='conv2',ismaxpool=True)
    flatted=flatten_layer(conv2)
    fc1=fc_layer_02(x=flatted,num_units=120,name='fc1',use_relu=True)
    fc2=fc_layer_02(x=fc1,num_units=84,name='fc2',use_relu=True)
    logits=fc_layer_02(x=fc2,num_units=10,name='logits',use_relu=False)
    #
    with tf.variable_scope('Train'):
        with tf.variable_scope('Loss'):
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits),name='loss')
            tf.summary.scalar(name='sum_loss',tensor=loss,collections=None,family=None)
        with tf.variable_scope('Optimizer'):
            optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        with tf.variable_scope('Accuracy'):
            correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
            accuracy=tf.reduce_mean(tf.cast(x=correct_prediction,dtype=tf.float32,name='Accuracy'))
            tf.summary.scalar(name='sum_accuracy',tensor=accuracy,collections=None,family=None)
    #
    summary=tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    #
    tmp_time=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer_train=   tf.summary.FileWriter(os.path.join(logdir,'model01',tmp_time,'train'))
    writer_train.add_graph(sess.graph)
    #
    writer_valid=   tf.summary.FileWriter(os.path.join(logdir,'model01',tmp_time,'valid'))
    writer_valid.add_graph(sess.graph)
    #
    writer_test=    tf.summary.FileWriter(os.path.join(logdir,'model01',tmp_time,'test'))
    writer_test.add_graph(sess.graph)
    #
    num_iteration=int(len(y_train)/batch_size)+1
    # training
    for epoch in range(epochs):
        print(f'Training epoch:\t{epoch+1}/{epochs}')
        # step 01: randomization train dataset
        X_train2,y_train2=shuffle(X_train,y_train)
        #
        for iteration in range(num_iteration):
            # calculate batch size
            global_step+=1
            start=iteration*batch_size
            end=(iteration+1)*batch_size
            X_batch=X_train2[start:end]
            y_batch=y_train2[start:end]
            
            feeddict_batch={x:X_batch,y:y_batch}
            #Optimizing training (backforward)
            sess.run(optimizer,feed_dict=feeddict_batch)
            #
            if iteration % freq01 == 0:
                # Calculate, display and write to summary the batch loss and accuracy
                [accuracy_batch,loss_batch,summary_batch]=sess.run([accuracy,loss,summary],feed_dict=feeddict_batch)
                writer_train.add_summary(summary=summary_batch, global_step=global_step)
                print(f'Epoch {epoch+1}\t Train:\t gstep: {global_step}\t iter: {iteration}\t Acc: {accuracy_batch}\t Loss: {loss_batch}')
            if iteration % freq02 == 0:
                # save the model checkpoint per 500 iteration
                saver.save(sess,os.path.join(logdir, f"checkpoint/model_checkpoint_{iteration}"))          
                
        # Run validation after every epoch
        [accuracy_valid,loss_valid, summary_valid]=  sess.run([accuracy,loss,summary],feed_dict={x:X_valid, y:y_valid})
        [accuracy_test, loss_test,  summary_test]=   sess.run([accuracy,loss,summary],feed_dict={x:X_test,  y:y_test})
        writer_valid.add_summary(   summary=   summary_valid,   global_step=global_step)
        writer_test.add_summary(   summary=    summary_test,   global_step=global_step)
        print('---'*10)
        print(f'Epoch {epoch+1}\t Valid:\t gstep: {global_step}\t iter: {iteration}\t Acc: {accuracy_valid}\t Loss: {loss_valid}')
        print(f'Epoch {epoch+1}\t test:\t gstep: {global_step}\t iter: {iteration}\t Acc: {accuracy_test}\t Loss: {loss_test}')
        print('---'*10)
        if accuracy_valid>stopping_accuracy:
            print('breaking')
            break
    # save the final model checkpoint when finish traininng
    saver.save(sess=sess,save_path=os.path.join(logdir,'checkpoint/model_checkpoint_final'))
    print('Training Done')  
    temp_path=os.getcwd()+'\\logdir'
    print(f'tensorboard --logdir={temp_path}')
    sess.close()


# In[6]:


# Training
model()


# In[ ]:




