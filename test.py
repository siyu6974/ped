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

clf = joblib.load('m_model.pkl') 

def non_max_suppression(area_list, geometry_info):
    print(geometry_info)
    for area in area_list:
# check if the centre of a window is in an area        
        m_x = (geometry_info[0]+geometry_info[0]+geometry_info[2])/2
        m_y = (geometry_info[1]+geometry_info[1]+geometry_info[3])/2
        if area["x1"]<m_x and area["x2"]>m_x and area["y1"]<m_y and area["y2"]<m_y:
            area["count"] += 1
        #TODO compare score
            return area_list
    m_area = {"x1":geometry_info[0], "x2":(geometry_info[0]+geometry_info[2]), "y1":geometry_info[1], "y2": (geometry_info[1]+geometry_info[3]), "count" : 0}
    area_list.append(m_area)
    return area_list


def sliding_window(img):
    test_windows = []
    label_list = []
    area_list = []
    max_height, max_width, deep= img.shape
    m_ratio_array = numpy.linspace(0.2, 0.8, num = 20)
    m_posotion_array = numpy.linspace(0, 0.8, num = 20)
    for m_ratio in m_ratio_array:
        height = int(max_height*m_ratio)
        width = int(max_width*m_ratio)
        
        for m_x_position in m_posotion_array:
            x = int(m_x_position * width)
            
            for m_y_position in m_posotion_array:
                y = int(m_y_position * height)
                label_list.append([x, y, width, height])
                cropped_img = img[y:min(y+height,max_height), x:min(x+width,max_width)]
                resized_img = resize(cropped_img, (48, 48))
                test_windows.append(resized_img.reshape(-1))
    sum = 0   
    i = 0
    print(clf.predict(test_windows))
    for pred in clf.predict(test_windows):
        if pred == 1:
            geometry_info = label_list[i]
            area_list = non_max_suppression(area_list, geometry_info)
            sum += 1
        else:
            sum = 0
#       if sum > 4:
#            print(label_list[i])
        i += 1
        
    print(area_list)

def main():
    
    train_image_list = []

    for filename in sorted(glob.glob('projetpers/test/*.jpg')): #assuming gif
        im=io.imread(filename)
        sliding_window(im)
        cropped_img = im[113:640, 215:484 ]
        io.imshow(cropped_img)
        plt.show()
        print("Debug")
        
if __name__ == '__main__':
   main()


#the ratio of sliding window is height : width = 1 : 2.3

    

