#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:46:43 2020

@author: shalini

Utility functions to segment the BG from an image using existing masks.
"""

import cv2
import numpy as np
import glob
from scipy import ndimage

input_img_folder = " " # Input input images folder

input_mask_folder = " " # Input seg mask folder

output_img_folder = " " # Output segmented images folder

def seg_using_gt_mask(img, gt_mask):

    gt_mask= remove_small_components(gt_mask[:, :, 0])
    #print(gt_mask.shape)
    gt_mask = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2)
    #print(gt_mask.shape)
    #gt_mask = gt_mask/255
    #mask2 = np.where((gt_mask==0),0,1).astype('uint8')

    #binary_img, frgrnd= remove_mask_border_graphcut(gt_mask)
    binary_img = gt_mask > 0.5

    open_img = ndimage.binary_opening(binary_img)
    final = img*open_img

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

def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_names_ = [f.split("/")[-1][:-4] for f in img_files]
    print(img_names_[0])
    counter = len(img_files)
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".png"

        mask_src = input_mask_folder +  img_name + ".png"
        input_mask = cv2.imread(mask_src)

        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        final_img = seg_using_gt_mask(input_img, input_mask)


        #final_img = seg_using_gt_mask(input_img, input_mask)

        cv2.imwrite(img_dest, final_img) #final_img
        #cv2.imwrite(mask_dest, resized_mask)
        #cv2.imwrite(output_img_folder+"final.png", resized_img)
        print("Countdown.", counter)
        counter = counter - 1
        #assert False
    pass

main()
