#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:16:49 2018

@author: Siyu Zhang
"""


import numpy as np
from skimage.feature import hog
from skimage import color,exposure

symbols_to_keep = dir()

#img = X_train[85]
#_, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
#                    cells_per_block=(8, 8),block_norm='L2-Hys', visualise=True)
#
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
#ax1.axis('off')
#ax1.imshow(img, cmap=plt.cm.gray)
#ax1.set_title('Input image')
## Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#ax2.axis('off')
#ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#ax2.set_title('Histogram of Oriented Gradients')
#plt.show()

def get_features(img):
    if img.shape[-1]==3:
        img = color.rgb2gray(img)
    return hog(img, orientations=9, pixels_per_cell=(8, 8), 
               cells_per_block=(8, 8),block_norm='L2-Hys', transform_sqrt=True)



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