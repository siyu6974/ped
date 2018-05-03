#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:16:49 2018

@author: Siyu Zhang
"""


import numpy as np
from skimage.feature import hog
from skimage import color
symbols_to_keep = dir()

def get_features(img):
    if img.shape[-1]==3:
        img = color.rgb2gray(img)
    return hog(img, orientations=8, pixels_per_cell=(8, 4), 
               cells_per_block=(2, 2),block_norm='L2-Hys', transform_sqrt=True)



tmp_X = np.zeros((X_train.shape[0], get_features(X_train[0]).shape[0]))

import concurrent.futures

tmp_X = concurrent.futures.ProcessPoolExecutor().map(get_features,X_train)
X_train = np.array(list(tmp_X))
    
# garbage collector
symbols_to_keep += ['X_train', 'Y_train', 'get_features']
internal_vars = ['name', 'internal_vars', 'symbols_to_keep', 'In', 'Out', 'get_ipython', 'exit', 'get_ipython', 'quit']
#
for name in dir():
    if not name.startswith('_') and name not in internal_vars:
        if name not in symbols_to_keep:
            del globals()[name]
del symbols_to_keep
del internal_vars
del name