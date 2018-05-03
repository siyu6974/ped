#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 23:05:04 2018

@author: aoao
"""


from skimage import io
import glob
import numpy

from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn import svm
from sklearn.externals import joblib

#clf = joblib.load('m_model.pkl') 
clf = model

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
          
#            if m_x1 < areas_list["x1"]:
 #               areas_list["x1"] = m_x1
  #          if m_x2 > areas_list["x2"]:
   #             areas_list["x2"] = m_x2
    #        if m_y1 < areas_list["y1"]:
     #           areas_list["y1"] = m_y1
      #      if m_y2 > areas_list["y2"]:
       #         areas_list["y2"] = m_y2

            areas_list["areas"].append(geometry_info)
        #TODO compare score
      
            return all_areas_list
        
    new_areas_list = {"x1":m_x1, "x2":m_x2, "y1":m_y1, "y2":m_y2}
    new_areas_list["areas"] = []
    new_areas_list["areas"].append(geometry_info)
    all_areas_list.append(new_areas_list)
    return all_areas_list


def sliding_window(img):
    H,W = SAMPLE_SIZE
    
    while img.shape[0]>=H and img.shape[1]>=W:
        x = y = 0
        while y+H <= img.shape[0] and x+W <= img.shape[1]:
            win = img[y:y+H, x:x+W]
            
            if clf.predict(win):
                geometry_info = [x,y,W,H]
                all_areas_list = non_max_suppression(all_areas_list, geometry_info)
            
        img = img

    
    all_areas_list = []
    max_height, max_width, deep= img.shape
    m_size_ratio_array = numpy.linspace(0.3, 0.05, num = 10)
    
    for m_ratio in m_size_ratio_array:
        
        width = int(max_width*m_ratio)
        height = int(width*2.3)
        
        m_posotion_ratio_array = numpy.linspace(0, 1-m_ratio, num = 10)
        
        label_list = []
        test_windows = []
        
        x = 0
        y = 0
        single_movement = int(width/8)
        while (x+width)<max_width:
            while  (y+height)<max_height:
                label_list.append([x, y, width, height])
                cropped_img = img[y:y+height, x:x+width]
                resized_img = resize(cropped_img, SAMPLE_SIZE)
                test_windows.append(resized_img.reshape(-1)) 
                y += single_movement
            x += single_movement
            
        i = 0
        for pred in clf.predict(test_windows):
            if pred == 1:
                geometry_info = label_list[i]
                all_areas_list = non_max_suppression(all_areas_list, geometry_info)     
            i += 1
    print(all_areas_list)
    
    
    

def main():

    for filename in sorted(glob.glob('projetpers/test/*.jpg')): #assuming gif
        print(filename)
        im=io.imread(filename)
#        io.imshow(cropped_img)
#        plt.show()
        sliding_window(im)
        
        
if __name__ == '__main__':
   main()


#the ratio of sliding window is height : width = 1 : 2.3

    

