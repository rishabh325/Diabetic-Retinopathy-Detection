import cv2
import os,sys
dirs = os.listdir('/home/radar/Documents/ml/MiniPrject/green/')
for img in dirs : 
	if img.endswith('.jpeg') :
		img2 = cv2.imread(img,-1)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		b,g,r=cv2.split(img2)
		cl_b = clahe.apply(b)
		cl_g = clahe.apply(g)
		cl_r = clahe.apply(r)
		cl_rgb = cv2.merge([cl_b,cl_g,cl_r]);
		cv2.imwrite('/home/radar/Documents/ml/MiniPrject/clahe_after_green/'+img,cl_rgb)