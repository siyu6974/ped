#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:26:30 2018

@author: laurent

"""

from skimage import io, util, color, feature
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm


train_pers_p = glob.glob('../imagepers/train/pos/*.png')
train_pers_p_float = np.zeros((800,30240))
train_pers_p_labels = np.ones(train_pers_p_float.shape[0])
j = 0
for path in train_pers_p:
    train_pers_p_float[j,:] = get_features(util.img_as_float(io.imread(path)))
    j = j+1
        
train_pers_n = glob.glob('../imagepers/train/neg/*.png')
train_pers_n_float = np.zeros((4001,30240))
train_pers_n_labels = np.zeros(train_pers_n_float.shape[0])

j = 0
for path in train_pers_n:
    train_pers_n_float[j,:] = get_features(util.img_as_float(io.imread(path)))
    j = j+1
    
train_pers_data = np.concatenate((train_pers_p_float, train_pers_n_float))
train_pers_labels = np.concatenate((train_pers_p_labels, train_pers_n_labels))



yp = model.predict(train_pers_data)
print(classification_report(train_pers_labels, yp))

mat = confusion_matrix(train_pers_labels,yp)
sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)
plt.xlabel('true')
plt.ylabel('predicted')