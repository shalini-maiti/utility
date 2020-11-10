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
import random

input_img_folder = "/data3/datasets/mano_homo_fixed_brightness_new_feas_with_bg_removed_23b/TRAIN/images/"
input_mask_folder = "/data3/datasets/mano_homo_fixed_brightness_new_feas_with_bg_removed_23b/TRAIN/masks/"

output_img_folder = "/data3/datasets/mano_homo_fixed_brightness_new_feas_with_bg_removed_tex_wrap_23c/TRAIN/images/"
imagenet_dir = glob.glob("/media/shalini/datapart/datasets/ILSVRC2012_img_test/*.JPEG") # Imagenet directory

def seg_using_gt_mask(img, gt_mask):   

    gt_mask = add_random_texture_to_mask(gt_mask)
    final = img*gt_mask
   
    #plt.figure()
    #io.imshow(final)
    #plt.figure()
    #io.imshow(final_grabcut)
    #plt.figure()
    #io.imshow(final)
    #io.show()
    #cv2.imwrite("masked.png", final)
    #print(gt_mask)
    #assert False
    return final

def remove_small_components(img):
    
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 250

    img_dst = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        #print(max(sizes) -sizes[i])
        if sizes[i] >= min_size:
            img_dst[output == i + 1] = 255
            

    return img_dst

def remove_mask_border(mask):
    if int(cv2.__version__[0]) == 3:
        _, contours, _ = cv2.findContours((mask[:, :, 0]).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif int(cv2.__version__[0]) == 4:
        contours, _ = cv2.findContours((mask[:, :, 0]).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_mask_2 = np.zeros_like(mask)
    cv2.drawContours(contour_mask_2, contours, 0, (255, 255, 255), 8)
    mask_single = mask.astype('float32') - contour_mask_2.astype('float32')
    indices = np.where(mask_single <= 0)
    mask_single[indices] = 0

    mask = mask_single.astype('uint8')
    return mask

def add_random_texture_to_mask(gt_mask):
    h, w, d = gt_mask.shape
    vert_col = send_valid_img()

    vert_col = cv2.resize(vert_col, (w,h)) # Resize to proper(Mano) dimensions
    vert_col = vert_col/255 # Normalize btwn 0 and 1
    
    #tex = np.random.rand(h, w, d)
    #mod_mask = gt_mask*np.array([random.uniform(0.01, 1), random.uniform(0.01, 1), random.uniform(0.01, 1)])
    mod_mask = gt_mask*vert_col
    mod_mask = mod_mask/255
    return mod_mask

def send_valid_img():
    invalid = True
    while invalid:
        vert_col_rand = random.sample(imagenet_dir, 1)
        vert_col = cv2.cvtColor(cv2.imread(vert_col_rand[0]), cv2.COLOR_BGR2RGB) # Read image and cvt to RGB
        print(vert_col.shape)
        invalid = all(np.less_equal(vert_col.mean(axis=0).mean(axis=0), np.array([122.5, 122.5, 122.5])))  
        print(vert_col.mean(axis=0).mean(axis=0))
        print(vert_col.shape)
        if(invalid is False):
            break
    return vert_col
def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_names_ = [f.split("/")[6][:-4] for f in img_files]
    print(img_names_[0])
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".png"
        
        mask_src = input_mask_folder +  img_name + ".jpg"
        input_mask = cv2.imread(mask_src)

        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        final_img = seg_using_gt_mask(input_img, input_mask)
        
        
        #final_img = seg_using_gt_mask(input_img, input_mask)

        cv2.imwrite(img_dest, final_img) #final_img
        #cv2.imwrite(mask_dest, resized_mask)
        #cv2.imwrite(output_img_folder+"final.png", resized_img)
        print("Fin.")
        #assert False
    pass

main()
    