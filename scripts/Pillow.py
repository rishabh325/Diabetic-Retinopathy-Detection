# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 04:37:40 2018

@author: radar
/home/radar/Documents/ml/MiniPrject/Pillow.py
"""

from PIL import Image
import os, sys
size_1=700,700;
size_2=512,512;

src = '/home/radar/Documents/ml/MiniPrject/original/'

dirs = os.listdir(src);

for img2 in dirs: 
    if not(os.path.isfile(src+img2)) or (not(img2.endswith('.jpeg'))):
        continue;
    img=Image.open(src+img2);
#img=Image.open('10_left.jpeg');
    img=img.resize(size_1, Image.ANTIALIAS);
    wd,ht=img.size;
    pix=img.load();
    for i in range(0,ht):
        for j in range(0,wd):
            if pix[j,i]>(15,15,15):
    #            print(pix[j,i])
                break;
        if pix[j,i]>(15,15,15):
            break;
    ht1=i;
#print(i);
    i=ht-1;
    while i>0:
        for j in range(0,wd):
            if pix[j,i]>(15,15,15):
                break;
        if pix[j,i]>(15,15,15):
            break;
        i=i-1;
    ht2=i;
#print(i);

    for i in range(0,wd):
        for j in range(0,ht):
            if pix[i,j]>(15,15,15):
                break;
        if pix[i,j]>(15,15,15):
            break;
    wd1=i;
    i=wd-1;
    while i>0:
        for j in range(0,ht):
            if pix[i,j]>(15,15,15):
                break;
        if pix[i,j]>(15,15,15):
            break;
        i=i-1;
    wd2=i;
    img3=img.crop((wd1,ht1,wd2,ht2));
    path='./';
    #print(img2);
    #f, e = os.path.splitext(path+img2)
    img3=img3.resize(size_2, Image.ANTIALIAS);
    img3.save('/home/radar/Documents/ml/MiniPrject/cropped_512*512/'+img2);
