#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:40:38 2020

@author: shalini
"""
import os
import sys
import glob
import shutil
import numpy as np
import random
import json
import time

total_number = 100000
#total_number = 620
number_of_datasets = 4
ratio = np.ones((number_of_datasets, ))
ratio = ratio/number_of_datasets

dataset_images_addrs = ['/data3/datasets/mano_homogeneous_vert_colour/mano_json_729/TRAIN/',
                        '/data3/datasets/mano_imagenet_vert_colour_2/TOTAL/images/'
                        '/data3/datasets/mano_arm_skin_color_8b_feasible_complete/TOTAL/images/', 
                        '/data3/datasets/mano_arm_imagenet_vert_colour_9_b_complete/TOTAL/images/'
                        ]
dataset_labels_addrs = ['/data3/datasets/mano_homogeneous_vert_colour/mano_json_729/TRAIN/',
                        '/data3/datasets/mano_imagenet_vert_colour_2/TOTAL/labels/',
                        '/data3/datasets/mano_arm_skin_color_8b_feasible_complete/TOTAL/labels/',
                        '/data3/datasets/mano_arm_imagenet_vert_colour_9_b_complete/TOTAL/labels/'
                        ]
output_images_addr =  '/data3/datasets/mano_imagenet_homo_vert_all_mixed_arm_14/TOTAL/images/'
output_labels_addr = '/data3/datasets/mano_imagenet_homo_vert_all_mixed_arm_14/TOTAL/labels/'


ratio_of_division = ratio*total_number
print(ratio_of_division)
counter = 0

for idx, dset_path in enumerate(dataset_labels_addrs):
   
    json_all = [f for f in glob.glob(dset_path + "*.json")]
    print('size',len(json_all))
    print('ration', ratio_of_division[idx])
    print(len(json_all))
    json_files = random.sample(json_all, int(ratio_of_division[idx]))
    img_names_ = [f.split("/")[6][:-5] for f in json_files]
    print("dset_path", dset_path)
      
    for img_name in img_names_:
        counter = counter + 1
        print(counter)
        #print(image_files_path + img_name + ".png")
        src1 = dataset_images_addrs[idx] + img_name + ".png"
        dest1 = output_images_addr + '/' + str(idx)+ img_name + ".png"
        #print(json_files_path + img_name + ".json")
        src2 = dataset_labels_addrs[idx] + img_name + ".json"
        print(idx == 0)
        '''
        if(idx == 0):
            with open(src2, 'r') as f:
                time.sleep(1)
                dat = json.load(f)
                print(dat)
                pts2DHand = np.array(dat['hand_pts'], dtype='f')
                pts2DHand = pts2DHand[:,[1,0]]
                dat['hand_pts'] = pts2DHand.tolist()
                
            tmp = dat
            
            with open(src2, 'w') as f:
                json.dump(tmp, f)
        '''
              
        dest2 = output_labels_addr + '/' + str(idx) + img_name + ".json"
        print("Img name", img_name)
        
        if os.path.exists(src1):
            #shutil.move(src1, dest1)
            #shutil.move(src2, dest1)
            shutil.copy(src1, dest1)
            shutil.copy(src2, dest2)
            
    