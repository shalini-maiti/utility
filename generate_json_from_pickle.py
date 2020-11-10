#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:04:09 2020

@author: shalini
"""

import glob
import json
import numpy as np

input_label_folder = "/data3/Segmentation/images/releaseTestSinisaBeta/CPMHand/Results_hand/0/"
output_json_folder = "/data3/Segmentation/images/releaseTestSinisaBeta/CPMHand/Results_hand/0_/"

def convert_to_json(label_arr, dest):

    label_dict = {}
    json_file = dest
    print(json_file)
    #print(len(sp[1:]))
    label_dict['hand_pts'] = (np.array(label_arr['KPS2D']).reshape(21,2)).tolist() # Invert y, x to x, y
    
    label_dict['is_left'] = 0
    g = open(json_file, 'w')
    json.dump(label_dict, g)
    return label_dict
    
def main():
    lables_input_files = [f for f in glob.glob(input_label_folder + "*.pickle")]
    label_names_ = [f.split("/")[8][:-7] for f in lables_input_files]
    print(label_names_[0])
    for label_name in label_names_:
        label_src = input_label_folder + label_name + ".pickle"
        input_label = np.load(label_src, allow_pickle=True)
        label_dest = output_json_folder + label_name + ".json"
        lable = convert_to_json(input_label, label_dest)
    pass
    
main()