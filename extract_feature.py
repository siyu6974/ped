#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:16:49 2018

@author: Siyu Zhang
"""


import numpy as np
from skimage.feature import hog


tmp_X = np.zeros((X_train.shape[0], 133920))

for i,d in enumerate(X_train):
    fd = hog(d, orientations=8, pixels_per_cell=(4, 4), 
                    cells_per_block=(3, 3),block_norm='L2-Hys', transform_sqrt=True)
    tmp_X[i] = fd
    
X_train = tmp_X
    
# garbage collector
vars_to_keep = ['X_train', 'Y_train']
internal_vars = ['name', 'internal_vars', 'vars_to_keep', 'In', 'Out', 'get_ipython', 'exit', 'get_ipython', 'quit']
#
for name in dir():
    if not name.startswith('_') and name not in internal_vars:
        if name not in vars_to_keep:
            del globals()[name]
del vars_to_keep
del internal_vars
del name