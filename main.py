#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:44:17 2018

@author: Siyu Zhang
"""


SAMPLE_SIZE = (128,64)


runfile('get_training_data.py') 
runfile('extract_feature.py')
#runfile('train.py')
runfile('test.py')
