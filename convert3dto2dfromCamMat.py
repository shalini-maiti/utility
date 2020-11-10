#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:03:53 2020

@author: shalini
"""
import os
import json
import numpy as np
import png
import cv2
import glob
from scipy import ndimage
import scipy.io


rot_mat= cv2.Rodrigues(np.array([0.00531, -0.01196, 0.00301]))[0]
#rot_mat= np.eye(3)
#trans= [24.0381,  0.4563,   1.2326]
trans= [-24.0381,  -0.4563,   -1.2326]
#trans=[0., 0., 0.]
c=[314.78337, 236.42484]
f= [607.92271, 607.88192]
#f= [475.62768, 474.77709]
#c=[336.41179, 238.77962]
#c = [318.47345, 250.31296]
#f = [822.79041, 822.79041]
input_labels = "/data3/datasets/STB/labels/B1Counting_SK.mat"
input_img_folder = "/data3/datasets/STB/B1Counting/"
output_labels_folder = "/data3/datasets/STB/B1CountingLabels/"
img_filename_filter = '/data3/datasets/STB/B1Counting/SK_color_{}.png'

def project_3d_to_2d(rt, t_vec, f, global_coord, c=np.zeros((2,1))):
    pixel_coord = np.zeros((3,1))
    K = np.array([[f[0], 0, c[0]],
                  [0, f[1], c[1]],
                  [0, 0, 1]])
    rot_trans = np.column_stack((rt, t_vec))
    #print(rot_trans)
    rot_trans = np.row_stack((rot_trans, np.array([0, 0, 0, 1]).reshape(1, 4)))
    #print(rot_trans)
    #assert False
    #rot_trans = np.eye(4)
    global_coord = np.append(global_coord, 1)
    #print("K:" + str(K.shape) + "rot_trans" + str(rot_trans.shape) + "global_coord" + str(global_coord))
    #print(global_coord)
    #print(rot_trans)
    #print(K)
    #pixel_coord = np.matmul(K, np.matmul(rot_trans, global_coord))
    #print(pixel_coord)

    pixel_coord = np.dot(np.linalg.inv(rot_trans), global_coord)
    #print(pixel_coord.shape)
    pixel_coord = pixel_coord.reshape(4, 1)[:3, :]
    pixel_coord = np.dot(K, pixel_coord)
    
   # print("pixel_coord", str(pixel_coord))
    x_2d = pixel_coord[0] if pixel_coord[2] == 0 else pixel_coord[0]/pixel_coord[2]
    y_2d = pixel_coord[1] if pixel_coord[2] == 0 else pixel_coord[1]/pixel_coord[2]
    #print(x_2d, y_2d)
    #assert False
    
    return x_2d, y_2d

def save_to_json(pos_for_labelling, plot_counter, labelDir):
    label_dict = {}
    json_file = 'SK_color_' + str(plot_counter) + '.json'
    
    print(json_file)
    #print(len(sp[1:]))
    label_dict['hand_pts'] = (pos_for_labelling).tolist()
    
    label_dict['is_left'] = 0
    g = open(labelDir + json_file, 'w')
    json.dump(label_dict, g)
    pass

def cal_wrist_coord(keypoint_uv21):
    '''
    wrist_vis = np.logical_or(keypoint_vis21[16], keypoint_vis21[0])
    keypoint_vis21 = np.concat([np.expand_dims(wrist_vis, 0),
                                keypoint_vis21[1:]], 0)
    '''
    wrist_uv = keypoint_uv21[16, :] + 1.5*(keypoint_uv21[0, :] - keypoint_uv21[16, :])
    keypoint_uv21 = np.concatenate([np.expand_dims(wrist_uv, 0),
                               keypoint_uv21[1:, :]], 0)
    return  np.around(keypoint_uv21).astype(int)
def main():
    #img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_files = glob.glob(input_img_folder + "*color*.png")
    img_files.sort()
    #img_names_ = [f.split("/")[5][:-4] for f in img_files]
    mat = scipy.io.loadmat(input_labels)
    hand_details = mat["handPara"]
    pos_arr = np.zeros((21,2))
    print(hand_details.shape)
    for i in range(hand_details.shape[2]):
        #img_ = [s for s in img_files if 'color_'+str(i)+'.' in s]
        img_filename = img_filename_filter.format(i)
        #img_data = cv2.imread(img_files[i])
        print(img_filename)
        img_data = cv2.imread(img_filename)

        #assert False
        coords_3d = hand_details[:, :, i]
        
        for coord in range(coords_3d.shape[1]):
          #print(coord)
          pos_arr[coord, 0], pos_arr[coord, 1] = project_3d_to_2d(rt=np.array(rot_mat), t_vec=trans, 
                                               f=np.array(f), global_coord=coords_3d[:, coord], c=np.array(c))
          pos_for_circle = np.around(pos_arr)
          pos_for_circle = pos_for_circle.astype(int)
          cv2.circle(img_data, (pos_for_circle[ coord, 0], pos_for_circle[coord, 1]), 5, (0, 255, 0), thickness=1, lineType=8, shift=0)
        

        new_pos = cal_wrist_coord(pos_arr)
        #cv2.circle(img_data, (new_pos[ 0, 0], new_pos[0, 1]), 5, (0, 255, 0), thickness=1, lineType=8, shift=0)
        #cv2.circle(img_data, (new_pos[ 16, 0], new_pos[16, 1]), 5, (0, 255, 0), thickness=1, lineType=8, shift=0)
        print(new_pos.shape)
        cv2.imshow("image", img_data)
        cv2.waitKey(1)
        save_to_json(new_pos, i, output_labels_folder)
        #assert False
main()