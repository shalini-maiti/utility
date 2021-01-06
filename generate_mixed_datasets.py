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
number_of_datasets = 5
ratio = np.ones((number_of_datasets, ))
ratio = ratio/number_of_datasets

'''
dataset_images_addrs = ['/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_1/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_2/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_3/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_4/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_5/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_6',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_7',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_8',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_9/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_10',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_11/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_12/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_13/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_14/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_15/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_16/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_17',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_18/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_19/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_20/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/augmented_samples/data_21/']

dataset_labels_addrs = ['/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_1/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_2/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_3/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_4/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_5/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_6/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_7/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_8/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_9/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_10/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_11/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_12/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_13/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_14/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_15/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_16/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_17/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_18/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_19/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_20/',
                        '/data3/datasets/multiview_hand_pose_dataset_v2/projections_2d/data_21/']


output_images_addr =  '/data3/datasets/multiview_hand_pose_dataset_v2/MultiviewHandPoseAugmentedAll/images'
output_labels_addr = '/data3/datasets/multiview_hand_pose_dataset_v2/MultiviewHandPoseAugmentedAll/labels'
'''
'''
dataset_images_addrs = ["/data3/datasets/STB/B1Counting/",
                        "/data3/datasets/STB/B2Counting/",
                        "/data3/datasets/STB/B3Counting/",
                        "/data3/datasets/STB/B4Counting/",
                        "/data3/datasets/STB/B5Counting/",
                        "/data3/datasets/STB/B6Counting/",]
dataset_labels_addrs = ["/data3/datasets/STB/B1CountingLabels/",
                        "/data3/datasets/STB/B2CountingLabels/",
                        "/data3/datasets/STB/B3CountingLabels/",
                        "/data3/datasets/STB/B4CountingLabels/",
                        "/data3/datasets/STB/B5CountingLabels/",
                        "/data3/datasets/STB/B6CountingLabels/"]
output_images_addr = "/data3/datasets/STB/allCountingImages" 
output_labels_addr = "/data3/datasets/STB/allCountingLabels"
'''
dataset_images_addrs = ["/data3/datasets/mano_like_24d_more_images_with_shape_24nd/TRAIN/images/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_24ne/TRAIN/images/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_with_tex_24nf/TRAIN/images/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_with_sp_24nm/TRAIN/images/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_with_blur_24nj/TRAIN/images/",
                        "/data3/datasets/mano_lighting_var_shape_param_bg_removed_27i/TRAIN/images/"
                        ]
dataset_labels_addrs = ["/data3/datasets/mano_like_24d_more_images_with_shape_24nd/TRAIN/labels/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_24ne/TRAIN/labels/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_with_tex_24nf/TRAIN/labels/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_with_sp_24nm/TRAIN/labels/",
                        "/data3/datasets/mano_like_24d_more_images_with_shape_base_with_blur_24nj/TRAIN/labels/",
                        "/data3/datasets/mano_lighting_var_shape_param_bg_removed_27i/TRAIN/labels/"
                        ]
output_images_addr = "/data3/datasets/mano_like_24d_more_images_with_shape_all_variatios_with_lighting_24ny/TRAIN/images"
output_labels_addr = "/data3/datasets/mano_like_24d_more_images_with_shape_all_variatios_with_lighting_24ny/TRAIN/labels"
ratio_of_division = ratio*total_number
print(ratio_of_division)
counter = 0
#image_per_ds = 200

for idx, dset_path in enumerate(dataset_images_addrs):
   
    json_all = [f for f in glob.glob(dset_path + "*.png")] # mostly .png STB
    print('size',len(json_all))
    #print('ration', ratio_of_division[idx])
    print(len(json_all))
    #json_files = random.sample(json_all, int(ratio_of_division[idx]))
    json_files = json_all
    #json_files = random.sample(json_all, image_per_ds)
    img_names_ = [f.split("/")[-1][:-4] for f in json_files]
    #img_names_ = [f.split("/")[-1][:-4] for f in json_files] # For STB dataset
				
    print("dset_path", dset_path)
      
    for img_name in img_names_:
        #img_name = [s for s in img_name.split("_") if s.isdigit()][0] # For STB dataset only, SK_01
        #img_name_num = [s for s in img_name.split("_") if s.isdigit()] # For MHP dataset only, 435_webcam_0
        # Remove above lines if the name is 1098_xxx
        
        counter = counter + 1
        print(counter)
        #print(image_files_path + img_name + ".png")
        src1 = dataset_images_addrs[idx] + img_name + ".png" # Other mano type, frei datasets
        
        
        #print(json_files_path + img_name + ".json")
        src2 = dataset_labels_addrs[idx] + img_name + ".json"
        print(idx == 0)
        
        #src1 = dataset_images_addrs[idx] +"SK_color_" + img_name + ".png" # For STB dataset
        #src2 = dataset_labels_addrs[idx] +"SK_color_" + img_name + ".json" # For STB dataset
        
        #src1 = dataset_images_addrs[idx] + img_name_num[0] + "_webcam_" + img_name_num[1] + ".jpg" # For MHP dataset
        #src2 = dataset_labels_addrs[idx] + img_name_num[0] + "_jointsCam_" + img_name_num[1] + ".json" # For MHP dataset
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
        #dest1 = output_images_addr + '/' +'0' + str(idx)+ img_name_num[0]+ img_name_num[1] + ".jpg" # MHP mostly .png Other datasets     
        #dest2 = output_labels_addr + '/' + '0' + str(idx)+ img_name_num[0]+ img_name_num[1] + ".json"
        
        dest1 = output_images_addr + '/' + '0'+ str(idx) + img_name + ".png" # For STB
        dest2 = output_labels_addr + '/' + '0'+ str(idx) + img_name + ".json" # For STB
        
        # Just in ann order to make it orderly
        #dest1 = output_images_addr + '/' + str(counter) + ".png" # For STB
        #dest2 = output_labels_addr + '/' + str(counter) + ".json" # For STB
        print("Img name", img_name)
        
        if os.path.exists(src1):
            #shutil.move(src1, dest1)
            #shutil.move(src2, dest1)

            shutil.copy(src1, dest1)
            shutil.copy(src2, dest2)
            #assert False
            pass               