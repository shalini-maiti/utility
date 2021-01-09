#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:55:48 2020

@author: shalini

Test out rotation, scale augmentation on images.
"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import json
import imutils


input_labels_folder =" " # Input labels folder
input_img_folder = " " # Input images folder

def rotAug(img, kps2D, minAng, maxAng, center=np.array([0, 0])):
    rotAng = random.randint(minAng, maxAng)

    theta = np.deg2rad(rotAng)
    tx = img.shape[1]/2
    ty = img.shape[0]/2

    S, C = np.sin(theta), np.cos(theta)

    # Rotation matrix, angle theta, translation tx, ty
    H = np.array([[C, -S, tx],
                  [S,  C, ty],
                  [0,  0, 1]])

    # Translation matrix to shift the image center to the origin
    r, c, d = img.shape
    pts2DHand = np.array(kps2D)


    img_rot = imutils.rotate(img, rotAng)
    labels_rot = np.array([rotate(x, [tx, ty], rotAng) for x in pts2DHand])

    return labels_rot, img_rot

def rotate(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def scaleAug(img, kps2d, minScale, maxScale):
    (h, w, d) = img.shape
    w = float(w)
    h = float(h)
    center = np.array([w/2, h/2])
    scaleVal = random.uniform(minScale, maxScale)
    print("scaleVal", scaleVal)
    scaled_img = cv2.resize(img, None, fx = scaleVal, fy = scaleVal, interpolation = cv2.INTER_CUBIC)
    (h_n, w_n, d_n) = scaled_img.shape
    center_new = np.array([w_n/2, h_n/2])
    scaled_2d = (kps2d + center)*scaleVal - center_new

    return scaled_2d, scaled_img

def read_json(j_file):
    with open(j_file, 'r') as f:
            #time.sleep(0.5)
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')

    return pts2DHand

def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_names_ = [f.split("/")[6][:-4] for f in img_files]
    print(img_names_[0])
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        label_src = input_labels_folder + img_name + ".json"
        pts2DHand = read_json(label_src)
        print(np.array(pts2DHand).shape)
        #scale_pts, scale_img, center = scaleAug(input_img, pts2DHand, 0.8, 1)
        (r, c, d) = input_img.shape
        center = np.array([c/2, r/2])
        rotPts2d, rot_Img = rotAug(input_img, pts2DHand, 0, 0, center)
        scaled_2d, scaled_img = scaleAug(rot_Img, rotPts2d, 1.0, 1.0)


        f, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(input_img, cmap=plt.cm.gray, interpolation='nearest')
        ax0.scatter(pts2DHand[:, 0], pts2DHand[:, 1], c='b', marker='o')
        ax1.imshow(rot_Img, cmap=plt.cm.gray, interpolation='nearest')
        ax1.scatter(rotPts2d[:, 0], rotPts2d[:, 1], c='b', marker='o')
        ax2.imshow(scaled_img, cmap=plt.cm.gray, interpolation='nearest')
        ax2.scatter(scaled_2d[:, 0], scaled_2d[:, 1], c='b', marker='o')
        plt.show()

        assert False

main()
