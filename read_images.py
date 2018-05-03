#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:45:47 2018

@author: aoao
"""


from skimage import io, util, color
import glob
import numpy as np
from sklearn.utils import shuffle
from skimage.transform import resize, rescale
from matplotlib import pyplot as plt

symbols_to_keep = dir()

train_image_list = []
train_positive_image_list = []
train_raw_negative_image_list = []
train_negative_image_list = []

for filename in sorted(glob.glob('projetpers/train/*.jpg')): #assuming gif
    im = io.imread(filename)
    im = color.rgb2gray(im)
    train_image_list.append(im)
    
with open('projetpers/label.txt') as f:
    lines = f.readlines()
    

RATIO = 2

       
        
def generate_neg_imgs(img, H, W, x1, x2, y1, y2):
    H,W = SAMPLE_SIZE
    
    while img.shape[0]>=H and img.shape[1]>=W:
        ystep = max(int((img.shape[0]-H) * 0.1),1)
        xstep = max(int((img.shape[1]-W) * 0.1),1)
        for y in range(0, img.shape[0] - H, ystep):
            for x in range(0, img.shape[1] - W, xstep):
                inter_area_ratio = ((x2-x)*(y2-y))/((x2-x1)*(y2-y1))
                if inter_area_ratio < 0.8:  
                    window = img[y:y + H, x:x + W]
                    train_negative_image_list.append(window)
        img = rescale(img, 0.8, mode='constant')



for i, tmp_img in enumerate(train_image_list):
    line = lines[i].split(" ")
    x = int(line[1])
    y = int(line[2])
    width = int(line[3])
    height = int(line[4])
    
#get the size of raw image
    max_height, max_width = tmp_img.shape   
# the center of a window    
    x_c = int((2*x+width)/2)
    y_c = int((2*y+height)/2)
    
    if height/width > RATIO:
        width = int(height/RATIO)
    else:
        height = int(RATIO*width)

    x1 = max(int(x_c - width/2), 0)
    y1 = max(int(y_c - height/2), 0)
    x2 = min(int(x_c + width/2), max_width)
    y2 = min(int(y_c + height/2), max_height)

    cropped_img = tmp_img[y1: y2, x1: x2]
    resized_img = resize(cropped_img, SAMPLE_SIZE)
    
    train_positive_image_list.append(resized_img)
    
    generate_neg_imgs(tmp_img, height, width, x1, x2, y1, y2)
    
#delete small negative image

for neg_img in train_raw_negative_image_list:
    height, width = neg_img.shape
    if height*width > SAMPLE_SIZE[0]*SAMPLE_SIZE[1]:
        resized_img = resize(neg_img, SAMPLE_SIZE)
        train_negative_image_list.append(resized_img)
        
        
    
y_positive = np.ones(len(train_positive_image_list))
y_negative = np.zeros(len(train_negative_image_list))
train_positive_image_list = np.array(train_positive_image_list)
train_negative_image_list = np.array(train_negative_image_list)
X_train = np.concatenate((train_positive_image_list, train_negative_image_list))
Y_train = np.concatenate((y_positive, y_negative), axis=0)

X_train, Y_train = shuffle(X_train, Y_train)



# garbage collector
symbols_to_keep += ['X_train', 'Y_train']
internal_vars = ['name', 'internal_vars','symbols_to_keep','In', 'Out', 'get_ipython', 'exit', 'get_ipython', 'quit']
#
for name in dir():
    if not name.startswith('_') and name not in internal_vars:
        if name not in symbols_to_keep:
            del globals()[name]
del symbols_to_keep
del internal_vars
del name