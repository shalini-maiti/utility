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
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label   
import skimage

input_img_folder = "/data3/datasets/FreiHAND_pub_v2/FreihandsEvalDataset/from_evaluation/FreihandsEvalFinalWoutObj/images/"
input_label_folder = "/data3/datasets/FreiHAND_pub_v2/FreihandsEvalDataset/from_evaluation/FreihandsEvalFinalWoutObj/labels/"
#input_mask_folder = "/data3/datasets/freihands_cleaned/masks/"

output_img_folder = "/data3/datasets/FreiHAND_pub_v2/FreihandsEvalDataset/from_evaluation/FreihandsEvalFinalWoutObj/resized_images/"
output_label_folder = "/data3/datasets/FreiHAND_pub_v2/FreihandsEvalDataset/from_evaluation/FreihandsEvalFinalWoutObj/resized_labels/"
#output_mask_folder = "/data3/datasets/freihands_cleaned/masks_pre/"

resultant_width = 640
resultant_height = 480


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


def remove_mask_border_graphcut(mask):
    ret, thresh = cv2.threshold(mask, 250, 255, 0)
    if int(cv2.__version__[0]) == 3:
        _, contours, _ = cv2.findContours((thresh[:, :, 0]).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif int(cv2.__version__[0]) == 4:
        contours, _ = cv2.findContours((thresh[:, :, 0]).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_mask_2 = np.zeros_like(mask)
    cv2.drawContours(contour_mask_2, contours, 0, (255, 255, 255), cv2.FILLED, 8)

    refine_border = np.zeros_like(mask)
    cv2.drawContours(refine_border, contours, 0, (255, 255, 255), cv2.FILLED,  8)

    mask_without_border = mask.copy()
    mask_without_border[mask_without_border > 0] = 1
    indices = np.where(contour_mask_2 > 0)
    mask_without_border[indices] = 2

    im_th = mask[:, :, 0]
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask_filled = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask_filled, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_floodfill_inv_big = remove_small_components(im_floodfill_inv)

    im_floodfill_inv = im_floodfill_inv - im_floodfill_inv_big
    im_floodfill_inv = im_floodfill_inv.astype('uint8')
    # Combine the two images to get the foreground.
    mask_foreground = im_th | im_floodfill_inv
    mask_foreground = mask_foreground | refine_border[:, :, 0]

    return mask_without_border, mask_foreground[:, :, np.newaxis]


def seg_using_gt_mask(img, gt_mask):   
    
    
    #gt_mask= remove_small_components(gt_mask[:, :, 0])
    #gt_mask = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2)
    #binary_img = gt_mask > 0.5
    #open_img = ndimage.binary_opening(binary_img)    
    #print(gt_mask.shape)
    #final = img*open_img
    #print(gt_mask.shape)
    #gt_mask = gt_mask/255
    #mask2 = np.where((gt_mask==0),0,1).astype('uint8')
    
    thresholded = gt_mask > 50
    labels = label(thresholded, connectivity=1)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(counts,unique )) # the 0 label is by default background so take the rest
    list_seg_Sort = sorted(list_seg, reverse=True)
    print(list_seg)
    largest=list_seg_Sort[0:5]
    labels_max=[(labels == largest_i[1]).astype(int) for largest_i in largest]


    
    #binary_img, frgrnd= remove_mask_border_graphcut(gt_mask)


    final = img*labels_max[1]
    final = np.uint8(final)
    '''
    mask_without_border, mask_foreground = remove_mask_border_graphcut(final)
    #mask_foreground = mask_foreground > 50
    #final_new = img*mask_without_border
    # Apply canny edge detector algorithm on the image to find edges
    edges = cv2.Canny(final, 100,200)
    _, contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours", len(contours))
    cv2.drawContours(final, contours, -1, (0, 255, 0), -1) #---set the last parameter to -1
    # Plot the original image against the edges
    plt.subplot(121), plt.imshow(final)
    plt.title('Original Gray Scale Image')
    plt.subplot(122), plt.imshow(edges)
    plt.title('Edge Image')
    '''
    '''
    f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
    ax0.imshow(labels_max[0], cmap=plt.cm.gray, interpolation='nearest')    
    ax1.imshow(final, cmap=plt.cm.gray, interpolation='nearest')
    ax2.imshow(final_new, cmap=plt.cm.gray, interpolation='nearest')
    ax3.imshow(mask_without_border, cmap=plt.cm.gray, interpolation='nearest')
    ax4.imshow(np.reshape(np.repeat(mask_foreground[:, :, np.newaxis], 3, axis=2),(480,640,3)), cmap=plt.cm.gray, interpolation='nearest')    
    plt.show()
    '''
    
    #plt.figure()
    #io.imshow(final)
    #plt.figure()
    #io.imshow(labels_max)
    #plt.figure()
    #io.imshow(final)
    #io.show()
    #cv2.imwrite("masked.png", final)
    #print(gt_mask)
    #assert False
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

def main():
    json_files = [f for f in glob.glob(input_label_folder + "*.json")]
    img_names_ = [f.split("/")[-1][:-5] for f in json_files]
    print(json_files[0])
    print(img_names_[0])
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".jpg"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".jpg"
        
        label_src = input_label_folder + img_name + ".json"
        label_dest = output_label_folder + img_name + ".json"
        
        #mask_src = input_mask_folder +  img_name + ".jpg"
        #mask_dest = output_mask_folder +  img_name + ".jpg"
        #input_mask = cv2.imread(mask_src)
        
        img_w, img_h, img_d = input_img.shape
        size_ratio_w = resultant_width/img_w
        size_ratio_h = resultant_height/img_h
        
        resized_img = change_size(input_img, size_ratio_w, size_ratio_h)
        #resized_mask = change_size(input_mask, size_ratio_w, size_ratio_h)
        
        pts2DHand = new_json_file(label_src, size_ratio_w, size_ratio_h, label_dest)
        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        #cv2.imshow("cv", input_img)
        #final_img = seg_using_gt_mask(resized_img, resized_mask)
        
        #final_img = seg_using_gt_mask(input_img, input_mask)

        #cv2.imwrite(img_dest, final_img) #final_img
        #cv2.imwrite(mask_dest, resized_mask)
        cv2.imwrite(output_img_folder+ img_name +".png", resized_img)
        
        print("Fin.")
        #assert False
    pass

main()