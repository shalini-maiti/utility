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

input_images_addr ='/data3/datasets/ShreyasDataset/others/Mahdi_HandBeta/segmentation/0/raw_seg_results_/'
output_images_addr =  '/data3/datasets/ShreyasDataset/others/Mahdi_HandBeta/segmentation/0/raw_seg_results/'


counter = 0

for idx, dset_path in enumerate(input_images_addr):
   
    img_all = [f for f in glob.glob(input_images_addr + "*.png")]
    print('size',len(img_all))
    
    #img_names_ = [f.split("/")[6][:-5] for f in json_files]
    img_names_ = [f.split("/")[9][:-4] for f in img_all]
    print("dset_path", dset_path)
      
    for img_name in img_names_:
        counter = counter + 1
        print(counter)
        #print(image_files_path + img_name + ".png")
        src1 = input_images_addr + img_name + ".png"
        dest1 = output_images_addr + img_name[6:] + ".png"
        #print(json_files_path + img_name + ".json")
        #src2 = dataset_labels_addrs[idx] + img_name + ".json"
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
              
        #dest2 = output_labels_addr + '/' + str(idx) + img_name + ".json"
        print("Img name", img_name)
        
        if os.path.exists(src1):
            #shutil.move(src1, dest1)
            #shutil.move(src2, dest1)
            shutil.copy(src1, dest1)
            #shutil.copy(src2, dest2)
            #assert False
    