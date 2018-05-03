#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:44:17 2018

@author: Siyu Zhang
"""


SAMPLE_SIZE = (128,64)


runfile('read_images.py')
runfile('extract_feature.py')

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import RandomizedPCA, PCA
from skimage.transform import rescale
import glob
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


z_scaler = StandardScaler()
pca = PCA(svd_solver='randomized', n_components=350, whiten=True, random_state=233)
svc = svm.SVC(class_weight='balanced', gamma=0.01, kernel='rbf', C=80)
model = make_pipeline(z_scaler, pca, svc)

s = cross_val_score(model,X_train,Y_train,n_jobs=4,cv=8)
print("Accuracy: %0.3f (+/- %0.3f)" % (s.mean(), s.std() * 2))

model.fit(X_train, Y_train)


def non_max_suppression(all_areas_list, geometry_info):
    """
    all_areas_list = [{x1,x2,y1,y2,[]}]
    geometry_info = [x,y,w,h]
    """
    m_x1 = geometry_info[0]
    m_y1 = geometry_info[1]
    m_x2 = geometry_info[0]+geometry_info[2]
    m_y2 = geometry_info[1]+geometry_info[3]
    
# the centre of the window
    m_x_center = (m_x1 + m_x2)/2
    m_y_center = (m_y1 + m_y2)/2
    
    for areas_list in all_areas_list:
        # check if the centre of a window is in an area         
        if areas_list["x1"]<m_x_center and areas_list["x2"]>m_x_center and areas_list["y1"]<m_y_center and areas_list["y2"]>m_y_center:
            areas_list["areas"].append(geometry_info)
        #TODO compare score
            return all_areas_list
        
    new_areas_list = {"x1":m_x1, "x2":m_x2, "y1":m_y1, "y2":m_y2}
    new_areas_list["areas"] = []
    new_areas_list["areas"].append(geometry_info)
    all_areas_list.append(new_areas_list)
    return all_areas_list

def sliding_window(img, scale_step=0.8):
    H,W = SAMPLE_SIZE
    scale = 1.0
    
    while img.shape[0]>=H and img.shape[1]>=W:
        ystep = max(int((img.shape[0]-H) * 0.05),1)
        xstep = max(int((img.shape[1]-W) * 0.05),1)
        for y in range(0, img.shape[0] - H, ystep):
            for x in range(0, img.shape[1] - W, xstep):
                window = img[y:y + H, x:x + W]
                yield (int(y/scale), int(x/scale), int(H/scale), int(W/scale)), window
        img = rescale(img, scale_step, mode='constant')
        scale = scale*scale_step
            
test_file_names = sorted(glob.glob('projetpers/test/*.jpg'))
im=io.imread(test_file_names[1])
geos, patches = zip(*sliding_window(im))
geos = list(geos)
geos.reverse()
patches = list(patches)
patches.reverse()

patches_hog = np.array([get_features(patch) for patch in patches])

labels = model.predict(patches_hog)


all_areas_list = []

geos = np.array(geos)
fig, ax = plt.subplots()
ax.imshow(im, cmap='gray')
ax.axis('off')
for y, x, h, w in geos[labels==1]:
    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
    all_areas_list = non_max_suppression(all_areas_list, (x,y,w,h))
plt.show()


fig, ax = plt.subplots()
ax.imshow(im, cmap='gray')
ax.axis('off')
for area in all_areas_list:
    x = area['x1']
    y = area['y1']
    w = area['x2'] - area['x1']
    h = area['y2'] - area['y1']
    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))