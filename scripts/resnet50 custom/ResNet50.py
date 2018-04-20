# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:44:02 2018

@author: mohit3101
"""
import numpy as np
#from keras import layers
from keras.layers import Dropout,Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,AveragePooling2D,MaxPooling2D
from keras.models import Model
from convolutional_block import convolutional_block
from identity_block import identity_block
from keras.initializers import glorot_uniform
from keras import regularizers
#import sys
#sys.setrecursionlimit(10000)
#import theano
#theano.config.optimizer="None"
def ResNet50(input_shape = (64, 64, 3), classes = 5):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=None))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2,2),name="avg_pool")(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(512, activation='relu', use_bias=True,name='fc5',kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None)(X)
    #X = Dense(512, activation='relu',name='fc5')(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu', use_bias=True,name='fc6',kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None)(X)
    #X = Dense(256, activation='relu',name='fc6')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax',name='fc7', kernel_initializer = 'glorot_uniform')(X)

    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model