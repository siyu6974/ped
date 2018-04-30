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


train_image_list = []
train_positive_image_list = []
train_negative_image_list = []

for filename in sorted(glob.glob('projetpers/train/*.jpg')): #assuming gif
    im=io.imread(filename)
    train_image_list.append(im)
    
with open('projetpers/label.txt') as f:
    lines = f.readlines()
    

for i in range(0, len(lines)):
    tmp_img = train_image_list[i]
    line = lines[i].split(" ")
    x = int(line[1])
    y = int(line[2])
    width = int(line[3])
    height = int(line[4])
    cropped_img = tmp_img[y:y+height, x:x+width]
    resized_img = resize(cropped_img, (256, 256))
    train_positive_image_list.append(resized_img.reshape(-1))

# create 3 negative images for each raw_image
 
    max_height, max_width, deep = tmp_img.shape
    
    
    cropped_img = tmp_img[y:, int((2*x+width)/2):]
    resized_img = resize(cropped_img, (256, 256))
    train_negative_image_list.append(resized_img.reshape(-1))
    
    
    cropped_img = tmp_img[0:x, 0:y]
    resized_img = resize(cropped_img, (256, 256))
    train_negative_image_list.append(resized_img.reshape(-1))
   
    cropped_img = tmp_img[y+height:, x+width:]
    resized_img = resize(cropped_img, (256, 256))
    train_negative_image_list.append(resized_img.reshape(-1))


#    io.imshow(cropped_img)
#   plt.show()
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