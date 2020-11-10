#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:13:15 2020

@author: shalini
"""
import json
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

input_img_folder = "/data3/datasets/mano_arm_imagenet_vert_colour_and_bg_10_b_complete/FreihandsWithoutBg_cleaned_unresized/images/"
input_labels_folder = "/data3/datasets/mano_arm_imagenet_vert_colour_and_bg_10_b_complete/FreihandsWithoutBg_cleaned_unresized/labels/"

input_pred ='/data3/datasets/mano_bg_hand_lighting_variation_without_rand_noise_wrap_27e/RESULTS/Finetune_27b27c27_27d27e/freihands_without_chunks/2DPCK.pickle'
acc_thresh = 34  # 34 for frei, 26 for shreyasDS
tar_w = tar_h = 300
inp_w = inp_h = 140

def scaleAug(kps2d, w, h):
    center = np.array([w/2, h/2])
    scaleVal = tar_w/w
    #scaled_img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_CUBIC)
    (h_n, w_n) = (w*scaleVal, h*scaleVal)
    center_new = np.array([w_n/2, h_n/2])
    scaled_2d = (kps2d + center)*scaleVal - center_new
    return scaled_2d

def read_json(j_file):
    with open(j_file, 'r') as f:
            #time.sleep(0.5)
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')

    return pts2DHand

def calc_3d_error(gt, predictions):
    projErrs = np.nanmean(np.linalg.norm(gt - predictions, axis=2), axis=1)
    #print((projErrs))
    meanError = np.sum(projErrs, axis=0) / len(projErrs)
    
    return meanError

def prediction_accuracy(gt, predictions):
    tick = time.time()
    accurate = 0
    for n in range(gt.shape[0]):
        accurate = (accurate + 1) if (np.nanmean(np.linalg.norm(gt[n] - predictions[n], axis=1), axis=0) < acc_thresh) else accurate # 34 for frei,26 for shreyasDS
    acc = accurate*100./gt.shape[0]
    tock = time.time()
    return acc, tock - tick

def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.jpg")]
    img_names_ = [f.split("/")[6][:-4] for f in img_files]
    input_img = cv2.imread(img_files[0])
    label_src = input_labels_folder + img_names_[0] + ".json"
    pts2DHand = read_json(label_src)
        
    input_pred_values = np.load(input_pred, allow_pickle=True)
    est_re = np.array([scaleAug(img_kp, inp_w, inp_h) for img_kp in input_pred_values["est"]])
    gt_re = np.array([scaleAug(img_kp, inp_w, inp_h) for img_kp in input_pred_values["gt"]])
    scaled_img = cv2.resize(input_img, (tar_w, tar_h), interpolation = cv2.INTER_CUBIC)
    
    '''
    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(input_img, cmap=plt.cm.gray, interpolation='nearest')
    ax0.scatter(pts2DHand[:, 0], pts2DHand[:, 1], c='b', marker='o')
    ax1.imshow(scaled_img, cmap=plt.cm.gray, interpolation='nearest')
    ax1.scatter(gt_re[0, :, 0], gt_re[0, :, 1], c='b', marker='o')
    ax2.imshow(scaled_img, cmap=plt.cm.gray, interpolation='nearest')
    ax2.scatter(est_re[0, :, 0], est_re[0, :, 1], c='b', marker='o')
    plt.show()    
    '''
    acc, t_taken = prediction_accuracy(input_pred_values["gt"], input_pred_values["est"])
    print("Acc", acc, t_taken)
    
    print(calc_3d_error(gt_re, est_re))
    print(calc_3d_error(input_pred_values["gt"],  input_pred_values["est"]))
    
    '''
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        label_src = input_labels_folder + img_name + ".json"
        pts2DHand = read_json(label_src)
        print(np.array(pts2DHand).shape)
        #scale_pts, scale_img, center = scaleAug(input_img, pts2DHand, 0.8, 1)
        (r, c, d) = input_img.shape
        center = np.array([c/2, r/2])
        #rotPts2d, rot_Img = rotAug(input_img, pts2DHand, -90, 90, center)
        scaled_2d, scaled_img = scaleAug(input_img, pts2DHand, 1.5, 2.0)
        


    
        assert False
    '''
main()