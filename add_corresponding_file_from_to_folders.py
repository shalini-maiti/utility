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

input_fil_dir='/data3/datasets/FreiHAND_pub_v2/FreihandsTrainData/images_with_obj/'
corresponding_file_dir = '/data3/datasets/FreiHAND_pub_v2/FreihandsTrainData/labels/'
output_folder = '/data3/datasets/FreiHAND_pub_v2/FreihandsTrainData/labels_w_obj/'

'''
input_fil_dir='/data3/datasets/freihands_cleaned/images/'
corresponding_file_dir = '/data3/datasets/freihands_cleaned/masks/'
output_folder = '/data3/datasets/freihands_cleaned/masks_/'
'''
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
        if os.path.exists(src1):
            print(count, src1)
            shutil.copy(src1, dest1)
            #shutil.move(src1, dest1)


'''
input_fil_dir='/data3/datasets/mano_arm_imagenet_vert_colour_9_b_complete/TOTAL/images/'
corresponding_file_dir = '/data3/datasets/mano_arm_imagenet_vert_colour_9_b_complete/TOTAL/labels/'
output_folder = '/data3/datasets/mano_arm_imagenet_vert_colour_9_b_complete/TOTAL/discarded/'

mini = 9000
maxi = 100

count = 0
for file in os.listdir(input_fil_dir):
    #if file.endswith('.jpg'):
    if file.endswith('.png'):
        is_valid = False
        fname = file
        count = count + 1
        print(count, fname)
        src1 = corresponding_file_dir + fname[:-4] +'.json'

        dest1 = output_folder
        
        src2 = input_fil_dir + fname # when removing json files
        #print(src1)
        img_ = cv2.cvtColor(cv2.imread(src2), cv2.COLOR_BGR2GRAY)
        #cv2.imwrite(dest1+fname, img_)
        #img_ = img_/255
        #img_ = cv2.imread()
        #print("not empty", np.count_nonzero(img_), 640*480)
        #is_valid = np.all(np.equal(img_, np.zeros(img_.shape)))
        is_valid = True if np.count_nonzero(img_) == 0 else False
        print(is_valid, np.count_nonzero(img_))
        print(src1)
        print(src2)
        mini = np.count_nonzero(img_) if np.count_nonzero(img_) < mini else mini
        maxi = np.count_nonzero(img_) if np.count_nonzero(img_) > maxi else maxi
        print("mini-maxi", mini, maxi)
        
        print(is_valid, os.path.exists(src1), os.path.exists(src2))
        #assert False
        if os.path.exists(src2) and is_valid:
            shutil.move(src1, dest1)
            shutil.move(src2, dest1) # When removing json files
            #assert False
'''