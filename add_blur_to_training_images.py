#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:46:43 2020

@author: shalini
"""

import cv2
import numpy as np
import glob
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage

input_img_folder = "/data3/datasets/mano_bg_hand_lighting_variation_without_bg_27b/TRAIN/images/"

output_img_folder = "/data3/datasets/mano_bg_hand_lighting_variation_without_bg_blur_27d/TRAIN/images/"

def blur_image(img):   
    final =  cv2.GaussianBlur(img,(5,5),0)
    return final


def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_names_ = [f.split("/")[6][:-4] for f in img_files]
    print(img_names_[0])
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".png"

        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        final_img = blur_image(input_img)
        
        
        #final_img = seg_using_gt_mask(input_img, input_mask)

        cv2.imwrite(img_dest, final_img) #final_img

        print("Fin.", img_name)
        #assert False
    pass

main()
    