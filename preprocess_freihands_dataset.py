#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Freihands dataset:
    
- Remove objects
- Resize to 640x480
- Replace coloured bg with black bg
Created on Wed Jul 15 15:43:54 2020

@author: shalini
"""

import cv2
import json
import numpy as np
import glob
from skimage import io

input_img_folder = "/data3/datasets/freihands_cleaned/images/"
input_label_folder = "/data3/datasets/freihands_cleaned/labels/"
input_mask_folder = "/data3/datasets/freihands_cleaned/masks/"

output_img_folder = "/data3/datasets/freihands_cleaned/cleanedButNotResized/"
output_label_folder = "/data3/datasets/freihands_cleaned/labels_pre/"
output_mask_folder = "/data3/datasets/freihands_cleaned/masks_pre/"

resultant_width = 640
resultant_height = 480

def seg_using_gt_mask(img, gt_mask):
    mask2 = np.where((gt_mask==0),0,1).astype('uint8')
    final = img*mask2
    #io.imshow(img)
    #io.show()
    #cv2.imwrite("masked.png", final)
    #print(gt_mask)
    return final

def change_size(input_img, ratio_w, ratio_h):
    image_ = cv2.resize(input_img,None,fx=ratio_w, fy=ratio_h, interpolation = cv2.INTER_CUBIC)
    return image_

def new_json_file(input_label_file, ratio_w, ratio_h, dest_file):
    with open(input_label_file, 'r') as f:
        dat = json.load(f)
        pts2DHand = np.array(dat['hand_pts'], dtype='f')
        pts2DHand[:,0] = ratio_w*pts2DHand[:,0]
        pts2DHand[:,1] = ratio_h*pts2DHand[:, 1]
        dat['hand_pts'] = pts2DHand.tolist()
        temp = dat
    g = open(dest_file, 'w')
    json.dump(temp, g)
    return pts2DHand

def main():
    json_files = [f for f in glob.glob(input_label_folder + "*.json")]
    img_names_ = [f.split("/")[5][:-5] for f in json_files]
    print(json_files[0])
    print(img_names_[0])
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".jpg"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".jpg"
        
        label_src = input_label_folder + img_name + ".json"
        label_dest = output_label_folder + img_name + ".json"
        
        mask_src = input_mask_folder +  img_name + ".jpg"
        mask_dest = output_mask_folder +  img_name + ".jpg"
        input_mask = cv2.imread(mask_src)
        
        img_w, img_h, img_d = input_img.shape
        size_ratio_w = resultant_width/img_w
        size_ratio_h = resultant_height/img_h
        
        resized_img = change_size(input_img, size_ratio_w, size_ratio_h)
        resized_mask = change_size(input_mask, size_ratio_w, size_ratio_h)
        pts2DHand = new_json_file(label_src, size_ratio_w, size_ratio_h, label_dest)
        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        #final_img = seg_using_gt_mask(resized_img, resized_mask)
        final_img = seg_using_gt_mask(input_img, input_mask)
        cv2.imwrite(img_dest, final_img) #final_img
        #cv2.imwrite(mask_dest, resized_mask)
        #cv2.imwrite(output_img_folder+"final.png", resized_img)
        print("Fin.")
        #assert False
    pass

main()