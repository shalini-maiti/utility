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


dataset_images_addrs = ['/data3/Segmentation/images/releaseTest0028/rgb/4/',
                        '/data3/Segmentation/images/releaseTest0028/rgb/5/',
                        '/data3/Segmentation/images/releaseTest0028/rgb/6/',
                        '/data3/Segmentation/images/releaseTest0028/rgb/7/',
                        '/data3/Segmentation/images/releaseTest0028/rgb/8/',
                        '/data3/Segmentation/images/releaseTest0028/rgb/9/',
                        '/data3/Segmentation/images/releaseTest0052/rgb/10/',
                        '/data3/Segmentation/images/releaseTest0052/rgb/11/',
                        '/data3/Segmentation/images/releaseTest0052/rgb/12/',
                        '/data3/Segmentation/images/releaseTest0052/rgb/13/',
                        '/data3/Segmentation/images/releaseTest0052/rgb/14/',
                        '/data3/Segmentation/images/releaseTesthand_test_jacket_60/rgb/15/',
                        '/data3/Segmentation/images/releaseTestMahdiBeta/rgb/16/',
                        '/data3/Segmentation/images/releaseTestMarkusBeta/rgb/17/',
                        '/data3/Segmentation/images/releaseTestNatashaHand/rgb/18/',
                        '/data3/Segmentation/images/releaseTestSinisaBeta/rgb/19/'
                        ]
dataset_labels_addrs = ['/data3/Segmentation/images/releaseTest0028/CPMHand/KPS2DStick_hand/4/',
                        '/data3/Segmentation/images/releaseTest0028/CPMHand/Results_hand/5/',
                        '/data3/Segmentation/images/releaseTest0028/CPMHand/Results_hand/6/',
                        '/data3/Segmentation/images/releaseTest0028/CPMHand/Results_hand/7/',
                        '/data3/Segmentation/images/releaseTest0028/CPMHand/Results_hand/8/',
                        '/data3/Segmentation/images/releaseTest0028/CPMHand/Results_hand/9/',
                        '/data3/Segmentation/images/releaseTest0052/CPMHand/Results_hand/10/',
                        '/data3/Segmentation/images/releaseTest0052/CPMHand/Results_hand/11/',
                        '/data3/Segmentation/images/releaseTest0052/CPMHand/Results_hand/12/',
                        '/data3/Segmentation/images/releaseTest0052/CPMHand/Results_hand/13/',
                        '/data3/Segmentation/images/releaseTest0052/CPMHand/Results_hand/14/',
                        '/data3/Segmentation/images/releaseTesthand_test_jacket_60/CPMHand/Results_hand/15/',
                        '/data3/Segmentation/images/releaseTestMahdiBeta/CPMHand/Results_hand/16/',
                        '/data3/Segmentation/images/releaseTestMarkusBeta/CPMHand/Results_hand/17',
                        '/data3/Segmentation/images/releaseTestNatashaHand/CPMHand/Results_hand/18/',
                        '/data3/Segmentation/images/releaseTestSinisaBeta/CPMHand/Results_hand/19/'
                        ]
output_images_addr =  '/data3/Segmentation/images/ho3d_varied/images'
output_labels_addr = '/data3/Segmentation/images/ho3d_varied/labels'


counter = 0

for idx, dset_path in enumerate(dataset_labels_addrs):
   
    json_all = [f for f in glob.glob(dset_path + "*.json")]
    print('size',len(json_all))
    print(len(json_all))
    #json_files = random.sample(json_all, int(ratio_of_division[idx]))
    json_files = json_all 
    img_names_ = [f.split("/")[8][:-5] for f in json_files]
    print("dset_path", dset_path)
      
    for img_name in img_names_:
        counter = counter + 1
        idx_name = idx + 5
        print(counter)
        #print(image_files_path + img_name + ".png")
        src1 = dataset_images_addrs[idx] + img_name + ".png"
        dest1 = output_images_addr + '/' + str(idx_name)+ img_name + ".png"
        #print(json_files_path + img_name + ".json")
        src2 = dataset_labels_addrs[idx] + img_name + ".json"
        #print(idx == 0)
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
              
        dest2 = output_labels_addr + '/' + str(idx_name) + img_name + ".json"
        print("Img name", img_name)
        print(os.path.exists(src1), os.path.exists(src2))
        if os.path.exists(src1):
            #shutil.move(src1, dest1)
            #shutil.move(src2, dest1)
            shutil.copy(src1, dest1)
            shutil.copy(src2, dest2)
            #assert False
            
            
    