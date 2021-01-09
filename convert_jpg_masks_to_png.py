#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Can generate noise using bad thresholding
Created on Tue Jul 28 13:04:44 2020

@author: shalini
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

input_mask_folder = " " # Input mask folder
output_mask_folder = " " # Output mask folder

threshold = 100

def clean_and_convert(mask):
    ret,thresh1 = cv2.threshold(mask,3,255,cv2.THRESH_BINARY)
    hist,bins = np.histogram(mask.ravel(), 1,[0,1])
    '''
    titles = ['Original Image', 'BINARY']
    images = [mask, thresh1]


    for i in range(2):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])


    plt.show()
    '''
    return thresh1

def main():
    mask_files = [f for f in glob.glob(input_mask_folder + "*.jpg")]
    mask_names_ = [f.split("/")[6][:-4] for f in mask_files]
    print(mask_names_[0])
    counter = 0
    for mask_name in mask_names_:
        mask_src = input_mask_folder + mask_name + ".jpg"
        input_mask = cv2.imread(mask_src)
        mask_dest = output_mask_folder + mask_name + ".png"
        output_mask = clean_and_convert(input_mask)

        #print("in", input_mask[150:200, 250:300, 0])
        #print("out", output_mask[150:200, 250:300, 0])
        cv2.imwrite(mask_dest, output_mask)
        counter = counter + 1
        print(counter)

main()