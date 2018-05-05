#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 00:05:27 2018

@author: Siyu Zhang
"""


import numpy as np
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from skimage.transform import rescale
import glob
from skimage import io
import concurrent.futures




def non_max_suppression(all_areas_list, geometry_info):
    """
    all_areas_list = [{x1,x2,y1,y2,[]}]
    geometry_info = [x,y,w,h]
    """
    m_x1 = geometry_info[0]
    m_y1 = geometry_info[1]
    m_x2 = geometry_info[0]+geometry_info[2]
    m_y2 = geometry_info[1]+geometry_info[3]
    s = geometry_info[4]
    
    # the centre of the window
    m_x_center = (m_x1 + m_x2)/2
    m_y_center = (m_y1 + m_y2)/2
    
    for areas_list in all_areas_list:
        # check if the centre of a window is in an area         
        if areas_list["x1"]<m_x_center and areas_list["x2"]>m_x_center and areas_list["y1"]<m_y_center and areas_list["y2"]>m_y_center:
            areas_list["areas"].append(geometry_info)
            return all_areas_list
        
    new_areas_list = {"x1":m_x1, "x2":m_x2, "y1":m_y1, "y2":m_y2, "s":s}
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
     
        
def test_with_file(file_num=0, output=None): 
    test_file_names = sorted(glob.glob('projetpers/test/*.jpg'))
    im=io.imread(test_file_names[file_num])
    geos, patches = zip(*sliding_window(im))
    geos = np.array(list(geos)) 
    
    tmp = concurrent.futures.ProcessPoolExecutor().map(get_features,patches)
    patches_hog = np.array(list(tmp))
    
    labels = model.predict(patches_hog)
    scores = model.decision_function(patches_hog)*-1
    
    all_areas_list = []
    
    geos = np.array(geos)
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    ax.axis('off')
    
    infos = np.hstack((geos,scores[:,None])) 

    idx = np.argsort(scores)
    infos = infos[idx]
    for y, x, h, w, s in infos[labels[idx]==1]:
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red',
                                   alpha=0.3, lw=2, facecolor='none'))
        all_areas_list = non_max_suppression(all_areas_list, (x,y,w,h,s))
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    ax.axis('off')
    for area in all_areas_list:
        x = area['x1']
        y = area['y1']
        w = area['x2'] - area['x1']
        h = area['y2'] - area['y1']
        s = area['s']
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red',
                                   alpha=0.3, lw=2, facecolor='none'))
        if output is not None:
            output.write(f'{file_num} {x} {y} {w} {h} {s}\n')
        

if 'model' not in globals(): 
    SAMPLE_SIZE = (128,64) 
    model = joblib.load('m_model.pkl')   
#
#with open("Output.txt", "w") as text_file: 
#    test_with_file(1, text_file) 
#    
with open("Output.txt", "w") as text_file: 
    for test_img_num in range(len(glob.glob('projetpers/test/*.jpg'))): 
        test_with_file(test_img_num, text_file) 