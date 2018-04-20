import numpy as np
import os
import time
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.models import load_model
#from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import pandas as pd
import random

PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    random.shuffle(img_list)
    img_list=img_list[0:3000]
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        img_path = data_path + '/'+ dataset + '/'+ img 
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #print('Input image shape:', x.shape)
        img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 5
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
'''
labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3
'''
data=pd.read_csv('trainLabels.csv')
data= np.array(data)
r,c =data.shape
mp={}
for i in range(0,r):
	mp[str(str(data[i][0])+'.jpeg')] = data[i][1]
i=0
src_path='./data/train001_CLAHE_299by299/'
dirs=os.listdir(src_path)
for img in img_list:
	if not(os.path.isfile(src_path+img)) or (not(img.endswith('.jpeg'))):
		continue
	labels[i]=(mp.get(img))
	i=i+1
names = ['NoDR','EarlyDR','ModerateDR','SevereDR','NPDR']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#using pre trained weights and fine tuning
image_input = Input(shape=(299, 299, 3))
model = InceptionV3(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

last_layer = model.output
#x= Flatten(name='flatten')(last_layer)
#x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
#x = Flatten()(x)
x = Dense(512, activation='relu',name='fc-1')(last_layer)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 5 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)
#out = Dense(5, activation='softmax', name='output_layer')(last_layer)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-6]:
	layer.trainable = False


custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))
#custom_resnet_model.save('ResNet50_only_classifier_trained.h5')
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))




