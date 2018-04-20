from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pickle



model = InceptionV3(weights='imagenet', include_top=False)
#pickle_out=open('/home/radar/Documents/ml/MiniPrject/models/V3Feature.pickle','wb')
#pickle.dump(model,pickle_out)
'''
pickle_in=open('/home/radar/Documents/ml/MiniPrject/models/vgg16.pickle','rb')
model=pickle.load(pickle_in)
'''
img_path = '/home/radar/Documents/ml/MiniPrject/clahed_tn_256/10_left.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print(np.shape(features))