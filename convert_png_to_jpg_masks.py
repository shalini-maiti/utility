#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:22:54 2020

@author: shalini
"""
import cv2
import glob

input_png = '/data3/datasets/mano_hand_green_bg_28/TRAIN/masks_without_shadow/'
output_jpg = '/data3/datasets/mano_hand_green_bg_28/TRAIN/masks_without_shadow_jpgs/'

img_files = [f for f in glob.glob(input_png + "*.png")]
img_names_ = [f.split("/")[6][:-4] for f in img_files]

for img_name in img_names_:
    img_src = input_png + img_name + ".png"
    input_img = cv2.imread(img_src)
    img_dest = output_jpg + img_name + ".jpg"
    cv2.imwrite(img_dest, input_img)