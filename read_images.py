#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:45:47 2018

@author: aoao
"""


from skimage import io
import glob
import numpy
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn import svm
from sklearn.externals import joblib


train_image_list = []
train_positive_image_list = []
train_negative_image_list = []

for filename in sorted(glob.glob('projetpers/train/*.jpg')): #assuming gif
    im=io.imread(filename)
    train_image_list.append(im)
    
with open('projetpers/label.txt') as f:
    lines = f.readlines()
    

sum_height = 0
sum_width = 0

for i in range(0, len(lines)):
    tmp_img = train_image_list[i]
    line = lines[i].split(" ")
    x = int(line[1])
    y = int(line[2])
    width = int(line[3])

    height = int(line[4])
    
    sum_width += width
    sum_height += height
    
    cropped_img = tmp_img[y:y+height, x:x+width]
    resized_img = resize(cropped_img, (48, 48))
    train_positive_image_list.append(resized_img.reshape(-1))

# create 3 negative images for each raw_image
 
    max_height, max_width, deep= tmp_img.shape
    
    
    cropped_img = tmp_img[y:, int((2*x+width)/2):]
    resized_img = resize(cropped_img, (48, 48))
    train_negative_image_list.append(resized_img.reshape(-1))
    
    
    cropped_img = tmp_img[0:x, 0:y]
    resized_img = resize(cropped_img, (48, 48))
    train_negative_image_list.append(resized_img.reshape(-1))
   
    cropped_img = tmp_img[y+height:, x+width:]
    resized_img = resize(cropped_img, (48, 48))
    train_negative_image_list.append(resized_img.reshape(-1))


#    io.imshow(cropped_img)
#   plt.show()
y_positive = numpy.ones(len(train_positive_image_list))
y_negative = numpy.zeros(len(train_negative_image_list))

X_train = train_positive_image_list + train_negative_image_list
Y_train = numpy.concatenate((y_positive, y_negative), axis=0)

Y_train = Y_train.tolist()
X_train, Y_train = shuffle(X_train, Y_train)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)

joblib.dump(clf, 'm_model.pkl') 


#clf = joblib.load('m_model.pkl') 

#the ratio of sliding window is height : width = 1 : 2.3




