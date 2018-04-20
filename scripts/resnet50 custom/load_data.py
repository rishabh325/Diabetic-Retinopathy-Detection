# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 09:27:58 2018

@author: mohit3101
"""

import numpy as np
#matplotlib inline
#import sys
#sys.setrecursionlimit(10000)
#import theano
#theano.config.optimizer="None"


from ResNet50 import ResNet50
#!/usr/bin/env python
from sklearn import cross_validation
#from sklearn.metrics import accuracy_score

import pandas as pd
#import tensorflow as tf
#from sklearn.linear_model import LogisticRegression as logreg
#from sklearn import svm
import os
import cv2
from one_hot import one_hot_matrix
import random
#from tensorflow.python.framework import ops


X = np.ones((35123,64,64,3))
y = np.ones((1,35123))
data = pd.read_csv('trainLabels.csv')

data = np.array(data)

r,c = data.shape

mp = {}

for i in range(0,r) :
	mp[str(str(data[i][0])+'.jpeg')] = data[i][1] 


i=0

src_path = './train001_64by64_CLAHE/'

dirs = os.listdir(src_path)
random.shuffle(dirs)
dirs=dirs[0:10000]
for img in dirs :
    if not(os.path.isfile(src_path+img)) or (not(img.endswith('.jpeg'))) :
        continue
    img_obj=cv2.imread(src_path+img)
    img_arr=np.array(img_obj)
    #print(img_arr.shape)
    #print (i)
    X[i][:]=img_arr
    y[0][i] =(mp.get(img))
    i = i+1
    
print(X.shape,y.shape)

y=y.T
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.20)
m_train=X_train.shape[0]
m_test=X_test.shape[0]
print(m_train)
print(m_test)
y=y.T



#X_train = X_train/255.
#X_test = X_test/255.
X_train_reshaped=np.reshape(X_train,(m_train,64*64*3))
X_test_reshaped=np.reshape(X_test,(m_test,64*64*3))
X_train_reshaped=X_train_reshaped-np.mean(X_train_reshaped,axis=1,keepdims=True)
X_test_reshaped=X_test_reshaped-np.mean(X_test_reshaped,axis=1,keepdims=True)
X_train_reshaped=(X_train_reshaped/(np.std(X_train_reshaped,axis=1,keepdims=True)))
X_test_reshaped=(X_test_reshaped/(np.std(X_test_reshaped,axis=1,keepdims=True)))
X_train=np.reshape(X_train_reshaped,(m_train,64,64,3))
X_test=np.reshape(X_test_reshaped,(m_test,64,64,3))


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
y_train = one_hot_matrix(y_train,C=5).T
y_test = one_hot_matrix(y_test,C=5).T
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
y_train=np.squeeze(y_train,axis=0)
y_test=np.squeeze(y_test,axis=0)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
print("Creating model")
model = ResNet50(input_shape = (64, 64, 3), classes = 5)
print("Created model")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Compiled model")
model.fit(X_train, y_train, epochs = 50, batch_size = 32)
print("fitted model")
preds = model.evaluate(X_test, y_test)
print("Evaluated model")
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
