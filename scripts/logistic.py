from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.linear_model import LogisticRegression as logreg
from sklearn import svm
import os
import cv2
import pickle
X = np.random.randn(1428,64*64*3)
y = np.random.randn(1428)
data = pd.read_csv('trainLabels.csv')

data = np.array(data)

r,c = data.shape

mp = {}

for i in range(0,r) :
	mp[str(str(data[i][0])+'.jpeg')] = data[i][1] 


i=0
src_path = '/home/radar/Documents/ml/MiniPrject/images_64*64/'
dirs = os.listdir(src_path)
for img in dirs :
	if not(os.path.isfile(src_path+img)) or (not(img.endswith('.jpeg'))) :
		continue
	img_obj = cv2.imread(src_path+img)
	img_arr = np.array(img_obj).reshape(-1)
	X[i]=img_arr
	y[i] = (mp.get(img))
	i = i+1
	#print(img)
	#feature = 
	#print(img_arr)

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)

#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
'''
clf = svm.SVC()
clf.fit(X_train,y_train)

'''
'''
pickle_in = open("/home/radar/Documents/ml/MiniPrject/models/svm.pickle","rb")
clf = pickle.load(pickle_in)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
'''
'''
pickle_out = open("/home/radar/Documents/ml/MiniPrject/models/svm.pickle","wb")
pickle.dump(clf,pickle_out)
'''