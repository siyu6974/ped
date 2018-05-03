#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:45:47 2018

@author: aoao
"""


from skimage import io
import glob
import numpy as np
from sklearn.utils import shuffle
from skimage.transform import resize
from matplotlib import pyplot as plt


train_image_list = []
train_positive_image_list = []
train_raw_negative_image_list = []
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
    
#get the size of raw image
    max_height, max_width, deep = tmp_img.shape   
# the center of a window    
    x_c = int((2*x+width)/2)
    y_c = int((2*y+height)/2)
    
    if height/width < 2.3:
        height = int(2.3*width)
    else:
        width = int(height/2.3)
    x1 = max(int(x_c - width/2), 0)
    y1 = max(int(y_c - height/2), 0)
    x2 = max(int(x_c + width/2), max_width)
    y2 = max(int(y_c + height/2), max_height)

    cropped_img = tmp_img[y:min(y+height, max_height), x: min(x+width, max_width)]
    resized_img = resize(cropped_img, (128, 64))
    train_positive_image_list.append(resized_img.reshape(-1))

# create 3 negative images for each raw_image

    cropped_img = tmp_img[y2:, x2:]
    train_raw_negative_image_list.append(cropped_img)  
      
    cropped_img = tmp_img[0:x1, 0:y1]
    train_raw_negative_image_list.append(cropped_img)

    cropped_img = tmp_img[0:x1, y2:]
    train_raw_negative_image_list.append(cropped_img)
        
    cropped_img = tmp_img[x2:, 0:y1]
    train_raw_negative_image_list.append(cropped_img)
    
    
#delete small negative image

for neg_img in train_raw_negative_image_list:
    height, width, deep= neg_img.shape
    if height*width > 250:
        
        resized_img = resize(neg_img, (128, 64))
        io.imshow(resized_img)
        plt.show()
        train_negative_image_list.append(resized_img.reshape(-1))
        
        
    
    

y_positive = np.ones(len(train_positive_image_list))
y_negative = np.zeros(len(train_negative_image_list))
train_positive_image_list = np.array(train_positive_image_list)
train_negative_image_list = np.array(train_negative_image_list)
X_train = np.concatenate((train_positive_image_list, train_negative_image_list))
Y_train = np.concatenate((y_positive, y_negative), axis=0)

X_train, Y_train = shuffle(X_train, Y_train)



# garbage collector
vars_to_keep = ['X_train', 'Y_train']
internal_vars = ['name', 'internal_vars','vars_to_keep','In', 'Out', 'get_ipython', 'exit', 'get_ipython', 'quit']
#
for name in dir():
    if not name.startswith('_') and name not in internal_vars:
        if name not in vars_to_keep:
            del globals()[name]
            print(name)
del vars_to_keep
del internal_vars
del name