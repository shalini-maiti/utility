#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:14:18 2020

@author: shalini
"""
import numpy as np
import png
import cv2
import glob
from scipy import ndimage

input_mask_folder = "/data3/Segmentation/images/releaseTest/segmentation/4/visualization/"
input_depth_folder = "/data3/datasets/ShreyasDataset/depth/4/"
output_mask_folder= "/data3/datasets/ShreyasDataset/CPMHand/segmentation_masks/4/"
input_img_folder = "/data3/datasets/ShreyasDataset/CPMHand/segmented_img/4_/"
output_img_folder = "/data3/datasets/ShreyasDataset/CPMHand/segmented_img/4/"

try:
    from itertools import imap
except ImportError:
    # Python 3...
    imap=map

def load_depth(path):
    # PyPNG library is used since it allows to save 16-bit PNG
    r = png.Reader(filename=path)
    im = np.vstack(imap(np.uint16, r.asDirect()[2])).astype(np.float32)#[:, ::-1]
        # Display the noise image
    #cv2.imshow('depth', im)
    #cv2.waitKey(0)
    
    return im

def seg_using_gt_mask(img, gt_mask):   
    
    '''
    thresholded = gt_mask > 50
    labels = label(thresholded, connectivity=1)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(counts,unique )) # the 0 label is by default background so take the rest
    list_seg_Sort = sorted(list_seg, reverse=True)
    print(list_seg)
    largest=list_seg_Sort[0:5]
    labels_max=[(labels == largest_i[1]).astype(int) for largest_i in largest]
    '''
    #print(sorted(img.flatten()))
    print("**************************************************************************8")
    #print(sorted(gt_mask[:,:, 2].flatten()))
    final = img*gt_mask
    
    return final

def gen_clean_mask(old_mask, new_mask):
    final = (old_mask>0)*(new_mask>0)
    #print(sorted(final[:,:, 1].flatten()))
    #final = np.repeat(final[:, :, np.newaxis], 3, axis=2)
    return final
def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_names_ = [f.split("/")[7][:-4] for f in img_files]
    print(img_names_[0])
    for img_name in img_names_:
        mask_src = input_mask_folder + img_name + "_prediction.png"
        input_mask = cv2.imread(mask_src)
        mask_dest = output_mask_folder + img_name + "_prediction.png"
        #img_dest_eg = output_img_folder + img_name + "_eq"+ ".png"
        #img_dest_adeg = output_img_folder + img_name + "_adeq"+ ".png"
        
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".png"
        depth_src = input_depth_folder +  img_name + ".png"
        
        #input_depth = cv2.imread(depth_src)
        depth_image = load_depth(depth_src)
        depth_image = depth_image < 700
        depth_image = depth_image > 0
        #print(sorted(depth_image.flatten()))
        depth_mask = (depth_image*255).astype('uint8')
        depth_mask = np.repeat(depth_mask[:, :, np.newaxis], 3, axis=2)
        #print(sorted(depth_mask.flatten()))
        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)

        final_mask = gen_clean_mask(input_mask, depth_mask)
        final_green_mask = final_mask[:, :, 1]
        #final_green_mask = final_green_mask/255.
        final_mask = np.repeat(final_green_mask[:, :, np.newaxis], 3, axis=2)
       
        #print(sorted(final_mask.flatten()))
        
        final_clean_img = seg_using_gt_mask(input_img, final_mask)
        
        #cv2.imshow("input", final_green_mask)
        cv2.imwrite(img_dest, final_clean_img)
        #cv2.imwrite(img_dest, final_img) #final_img
        #cv2.imwrite(img_dest_eg, img_eq) #final_img
        
        #cv2.imwrite(img_dest, img_ad_eq) #final_img
        
        cv2.imwrite(mask_dest, final_mask*255)
        #cv2.imwrite(output_img_folder+"final.png", resized_img)
        print("Fin.")
        #assert False
    pass

main()