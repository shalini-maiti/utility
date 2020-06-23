#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:28:05 2020

@author: shalini
"""
import os
import sys
import glob
import shutil
import numpy as np
import random

input_fil_dir='/data3/datasets/mano_homogeneous_vert_colour/mano_json_729/Freihands/images/'
corresponding_file_dir = '/data3/datasets/FreiHAND_pub_v2/training/mask/'
output_folder = '/data3/datasets/mano_homogeneous_vert_colour/mano_json_729/Freihands/mask/'

count = 0
for file in os.listdir(input_fil_dir):
    if file.endswith('.jpg'):
        fname = file
        count = count + 1
        print(count, fname)
        src1 = corresponding_file_dir + fname
        dest1 = output_folder
        if os.path.exists(src1):
            shutil.copy(src1, dest1)
