#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:28:05 2020

@author: shalini
"""
import os
import sys
import glob
import shutil
import numpy as np
import random
import cv2


# For freihands seg masks corresponding to the test images

input_fil_dir= ' ' # Address of the input image folder
corresponding_file_dir = ' ' # Address of the folder from where to select label files
output_folder = ' ' # Address of the output folder into which the corresponding file will be copie


count = 0
for file in os.listdir(input_fil_dir):
    #if file.endswith('.jpg'):
    if file.endswith('.png'):
        is_valid = False
        fname = file
        count = count + 1
        #print(count, fname)
        src1 = corresponding_file_dir + fname[:-4] + '.json'
        #src1 = corresponding_file_dir + fname

        dest1 = output_folder
        if os.path.exists(src1) and os.path.exists(dest1):
            print(count, src1)
            shutil.copy(src1, dest1)
            #shutil.move(src1, dest1)
