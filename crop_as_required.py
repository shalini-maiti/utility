import os
import sys
import glob
import shutil
import numpy as np
import random
import json
import time
import cv2
import numpy as np


# GLOBAL VARIABLES
input_img_folder = '/data3/datasets/000mano_full_arm_homo_vert_colour/TOTAL/images/'
input_label_folder='/data3/datasets/000mano_full_arm_homo_vert_colour/TOTAL/labels/'

output_img_folder ='/data3/datasets/mano_homo_vert_cropped_to_eightpercent_15/TOTAL/images/'
output_label_folder='/data3/datasets/mano_homo_vert_cropped_to_eightpercent_15/TOTAL/labels/'

crop_ratio = 0.7
resize_ratio = 1/crop_ratio
'''
def read_and_modify_json_file(input_label_file, ratio, output_label_file):
    print(input_label_file)
    with open(input_label_file, 'r') as f:
        dat = json.load(f)
        pts2DHand = np.array(dat['hand_pts'], dtype='f')
    
        data = pts2DHand
    return data
'''

def resized_json_file(input_label_file, ratio_w, ratio_h, dest_file):
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

def crop_image(input_img_file, crop_ratio, resize_ratio, output_img_file, input_label_file, output_label_file):
    input_img = cv2.imread(input_img_file)
    (h, w, d) = input_img.shape # Row height, column width
    cropped_w = int(np.ceil(crop_ratio*w))
    cropped_h = int(np.ceil(crop_ratio*h))
    print(cropped_w, cropped_h)
    with open(input_label_file, 'r') as f:
        dat = json.load(f)
        pts2DHandtoCrop = np.array(dat['hand_pts'], dtype='f')[0]
    # Get the egde cases
    left_bound=  0 if int(pts2DHandtoCrop[0] - cropped_w/2) < 0 else int(pts2DHandtoCrop[0] - cropped_w/2)
    right_bound= (w-1) if int(pts2DHandtoCrop[0]+ cropped_w/2)> (w-1) else int(pts2DHandtoCrop[0]+ cropped_w/2)
    top_bound =  0 if int(pts2DHandtoCrop[1] - cropped_h/2)< 0 else int(pts2DHandtoCrop[1] - cropped_h/2)
    bottom_bound=  (h-1) if int(pts2DHandtoCrop[1]+cropped_h/2)> (h-1) else int(pts2DHandtoCrop[1]+cropped_h/2)
    print(pts2DHandtoCrop, cropped_w, cropped_h, left_bound,right_bound,top_bound,bottom_bound, input_img.shape)
    cropped_img = input_img[top_bound : bottom_bound, left_bound: right_bound, :]
    #cropped_img = input_img
    
    print("cropped", cropped_img.shape)   
    resized_img =  cv2.resize(cropped_img, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_CUBIC)
    print("resized_img", resized_img.shape)
    pts2DHand = resized_json_file(input_label_file, resize_ratio, resize_ratio, output_label_file)
    
    #pts2DHand = read_and_modify_json_file(input_label_file, ratio, output_label_file)
    print(pts2DHand.shape)
    '''
    for row in range(pts2DHand.shape[0]):
        cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        cv2.circle(resized_img, (5, 5), 5, (0, 255, 0), thickness=1, lineType=8, shift=0)
        cv2.circle(resized_img, (10,10), 5, (255, 0, 0), thickness=1, lineType=8, shift=0)
    '''

    cv2.imwrite(output_img_file, resized_img)
    #cv2.waitKey(0)
    #assert False
    pass

def main():
    
    json_files = [f for f in glob.glob(input_label_folder + "*.json")]
    img_names_ = [f.split("/")[6][:-5] for f in json_files]
    
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        img_dest = output_img_folder + '/' + img_name + ".png"
        label_src = input_label_folder + img_name + ".json"
        label_dest = output_label_folder + '/' + img_name + ".json"
        
        crop_image(img_src, crop_ratio, resize_ratio, img_dest, label_src, label_dest)
    pass
 

main()