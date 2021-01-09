#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:12:37 2020

@author: shalini
"""

import numpy as np
import json
import glob
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import imutils

our_labels= " " # Data folder 1
freihands_no_chunks_labels = " " # Data folder 2
freihands_train_labels = " " # Data folder 3
shreyas_ds_labels = " " # Data folder 4
stb_random_labels = " " # Data folder 5
stb_counting_labels = " " # Data folder 6
mhp_10k_labels = " " # Data folder 7
mhp_10k_aug_labels = " " # Data folder 8
RS = 20150101

# If you need to rearrange the distribution of the points, do so here.
# In my case, the sequence of the joints is different for some datasets, so
# I have to rearrange those joints.

jointsMap = [0,
            13, 14, 15, 16,
            1, 2, 3, 17,
            4, 5, 6, 18,
            10, 11, 12, 19,
            7, 8, 9, 20]

multiviewJointsMap = [20,
                      17, 19, 18, 16,
                      1, 3, 2, 0,
                      5, 7, 6, 4,
                      13, 15, 14, 12,
                      9,11, 10, 8]

def prep_and_transform(our_json_files, frei_no_chunk_json_files, ds_json_files,
                      minScale, maxScale, minRot, maxRot,
                      stb_random_json_files, mhp_10k_json_files,
                      frei_train_json_files, stb_counting_json_files,
                      mhp_10k_aug_json_files, tsne_comp, input_img=np.zeros(480, 640)):
    all_label_data = []
    y = []
    js_stuff = []

    (r, c, d) = input_img.shape
    center = np.array([c/2, r/2])

    y=0
    for x in range(1):
        for j_file in our_json_files:
            with open(j_file, 'r') as f:
                #time.sleep(0.5)
                dat = json.load(f)
                pts2DHand = np.array(dat['hand_pts'], dtype='f')
                rotPts2d, rot_Img = rotAug(input_img, pts2DHand, minRot, maxRot, center)
                scaled_2d, scaled_img = scaleAug(rot_Img, rotPts2d, minScale, maxScale)

                '''
                f, (ax0, ax1, ax2) = plt.subplots(1, 3)
                ax0.imshow(input_img, cmap=plt.cm.gray, interpolation='nearest')
                ax0.scatter(pts2DHand[:, 0], pts2DHand[:, 1], c='b', marker='o')
                ax1.imshow(rot_Img, cmap=plt.cm.gray, interpolation='nearest')
                ax1.scatter(rotPts2d[:, 0], rotPts2d[:, 1], c='b', marker='o')
                ax2.imshow(scaled_img, cmap=plt.cm.gray, interpolation='nearest')
                ax2.scatter(scaled_2d[:, 0], scaled_2d[:, 1], c='b', marker='o')
                plt.show()
                assert False
                '''
                all_label_data.append(scaled_2d[jointsMap][0:]) # [1: ] To forget the wrist joint
                js_stuff.append([0])
                y = y + 1
                print(y)



    for j_file in frei_no_chunk_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[0:])  # [1: ] To forget the wrist joint
            js_stuff.append([1])

    for j_file in ds_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[0:])  # [1: ] To forget the wrist joint
            js_stuff.append([7])

    for j_file in stb_random_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[0:])  # [1: ] To forget the wrist joint
            js_stuff.append([2])


    for j_file in stb_counting_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[0:])  # [1: ] To forget the wrist joint
            js_stuff.append([6])


    for j_file in mhp_10k_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[multiviewJointsMap][0:])  # [1: ] To forget the wrist joint
            js_stuff.append([3])

    for j_file in frei_train_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[0:])  # [1: ] To forget the wrist joint
            js_stuff.append([4])

    for j_file in mhp_10k_aug_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[multiviewJointsMap][0:])  # [1: ] To forget the wrist joint
            js_stuff.append([5])

    all_label_data = np.array(all_label_data)
    all_label_data = all_label_data.reshape(all_label_data.shape[0], -1)

    print(all_label_data.shape)

    X = all_label_data
    y = np.array(js_stuff)
    #y = [0]*len(our_json_files) + [1]*len(frei_json_files)
    y = np.reshape(y, (all_label_data.shape[0], ))
    #y = np.concatenate(y, np.ones((len(freihands_images))))
    #print(X[0], y[0])
    print(X.shape, y.shape)
    tsneTransform = TSNE(n_components=tsne_comp, random_state=RS).fit_transform(X)
    print("Tsne", tsneTransform.shape)

    np.save("/data3/datasets/mano_bg_like_24_with_depth_convertible_masks_24d/tsnePlot2dModel24d-28.npy", tsneTransform)
    return tsneTransform, y

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
    #print("scaleVal", scaleVal)
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


def scatter_2d(x, colors, len_our_json_files):
    fig = plt.figure()
    ax = fig.add_subplot(111)


    ax.scatter(x[colors ==0, 0], x[colors == 0, 1], c='crimson', marker='o', label="Base dataset w/ shape variation (30k)") # Ours
    #ax.scatter(x[colors ==1 , 0], x[colors == 1, 1], c='turquoise', marker='^', label="Freihands Segmented Training Data (788)") #Frei
    #ax.scatter(x[colors ==2 , 0], x[colors == 2, 1], c='tomato', marker='*', label="STB Random (9k)") #HO3D
    #ax.scatter(x[colors ==3 , 0], x[colors == 3, 1], c='darkolivegreen', marker="4", label="Large-scale Multiview 3D Hand Pose (10k)") #STB
    #ax.scatter(x[colors ==4 , 0], x[colors == 4, 1], c='deepskyblue', marker='H', label="Freihands Train Data Without Objects (2721)") #STB
    #ax.scatter(x[colors ==5 , 0], x[colors == 5, 1], c='darkgray', marker='H', label="Large-scale Multiview 3D Hand Pose With Augmented BG (10k)") #STB
    ax.scatter(x[colors ==6 , 0], x[colors == 6, 1], c='mediumslateblue', marker='H', label="STB Counting (9k)") #STB
    #ax.scatter(x[colors ==7 , 0], x[colors == 7, 1], c='lightseagreen', marker='H', label="Ho-3D DS (267)") #STB

    #ax.scatter(x[len_our_json_files:-1, 0], x[len_our_json_files:-1, 1],  c='b', marker='^')
    #ax.scatter(x[0:len_our_json_files, 0], x[0:len_our_json_files, 1],  c='r', marker='o')

    #ax.set_xlim(-50, 50)
    #ax.set_ylim(-50, 50)
    #ax.set_zlim(-50, 50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()
    #ax.set_zlabel('Z Label')

    plt.show()
    return plt
def scatter_3d(x, colors, len_our_json_files):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    #ax.scatter(x[colors ==1 , 0], x[colors == 1, 1], x[colors ==1, 2], c='turquoise', marker='^', label="Freihands Segmented Training Data (788)") #Frei
    #ax.scatter(x[colors ==2 , 0], x[colors == 2, 1], x[colors ==2, 2], c='tomato', marker='*', label="STB Random Dataset (9k)") #HO3D
    #ax.scatter(x[colors ==3 , 0], x[colors == 3, 1], x[colors ==3, 2], c='darkolivegreen', marker="4", label="MultiviewHandPose (10k)") #STB
    #ax.scatter(x[colors ==4 , 0], x[colors == 4, 1], x[colors ==4, 2], c='deepskyblue', marker='H', label="Freihands Train Data Without Objects (2721)") #STB
    #ax.scatter(x[colors ==5 , 0], x[colors == 5, 1], x[colors ==5, 2], c='darkgray', marker='H', label="MultiviewHandPose With Augmented BG (10k)") #STB
    #ax.scatter(x[colors ==6 , 0], x[colors == 6, 1], x[colors ==6, 2], c='mediumslateblue', marker='H', label="STB Counting Dataset (9k)") #STB
    ax.scatter(x[colors ==7 , 0], x[colors == 7, 1], x[colors ==7, 2], c='lightseagreen', marker='H', label="Shreyas DS (267)") #STB
    ax.scatter(x[colors ==0, 0], x[colors == 0, 1], x[colors ==0, 2], c='crimson', marker='o', label="Base Dataset w/ shape variation (30k)") # Ours
    #ax.scatter(x[0:len_our_json_files, 0], x[0:len_our_json_files, 1], x[0:len_our_json_files, 2],  c='r', marker='o')
    #ax.scatter(x[len_our_json_files:-1, 0], x[len_our_json_files:-1, 1], x[0:len_our_json_files, 2], c='b', marker='^')
    #ax.scatter(x[len_our_json_files:-1, 0], x[len_our_json_files:-1, 1], x[0:len_our_json_files, 2], c='g', marker='-')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.show()
    pass

def main():
    minScale = 0.6
    maxScale = 1.4
    minRot= -180
    maxRot= 180
    our_json_files = [f for f in glob.glob(our_labels + "*.json")]
    tsne_comp = 2

    our_json_files = random.sample(our_json_files, 30000)
    frei_no_chunk_json_files = [f for f in glob.glob(freihands_no_chunks_labels + "*.json")]
    frei_train_json_files = [f for f in glob.glob(freihands_train_labels + ".json")]
    ds_json_files = [f for f in glob.glob(shreyas_ds_labels + "*.json")]
    stb_random_json_files = [f for f in glob.glob(stb_random_labels + "*.json")]
    stb_counting_json_files = [f for f in glob.glob(stb_counting_labels + "*.json")]
    mhp_10k_json_files = [f for f in glob.glob(mhp_10k_labels + "*.json")]
    mhp_10k_aug_json_files = [f for f in glob.glob(mhp_10k_aug_labels + "*.json")]

    '''
    tsne2d = np.load("/data3/datasets/mano_bg_fixed_all_with_arm_25/tsnePlot2d_freihandsVsModel24d.npy")
    y = [0]*len(our_json_files[0:10000]) + [1]*len(frei_json_files)
    print("Our Labels", min(tsne2d[0:len(our_json_files), 0]), max(tsne2d[0:len(our_json_files), 0]), min(tsne2d[0:len(our_json_files), 1]), max(tsne2d[0:len(our_json_files), 1]))
    print("Frei Labels", min(tsne2d[len(our_json_files):-1, 0]), max(tsne2d[len(our_json_files):-1, 0]), min(tsne2d[len(our_json_files):-1, 1]), max(tsne2d[len(our_json_files):-1, 1]))
    plt_2d = scatter_2d(tsne2d, y, len(our_json_files[0:10000]))
    '''
    '''
    freihands_img_folder = "/data/maiti/data3/datasets/mano_imagenet_homo_vert_mixed_5/FreihandsNew788_resized/images/"
    our_image_folder = "/data/maiti/data3/datasets/mano_homo_vert_imagenet_bg_16/TRAIN/images/"
    freihands_images =  [f for f in glob.glob(freihands_img_folder + "*.jpg")]
    our_images = [f for f in glob.glob(our_image_folder + "*.png")]
    '''

    #img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    #img_names_ = [f.split("/")[6][:-4] for f in img_files][20]

    tsneTransform, y = prep_and_transform(our_json_files, frei_no_chunk_json_files, ds_json_files,
                                              minScale, maxScale, minRot, maxRot,
                                              stb_random_json_files, mhp_10k_json_files,
                                              frei_train_json_files, stb_counting_json_files,
                                              mhp_10k_aug_json_files, tsne_comp)
    if tsne_comp == 2:
        scatter_2d(tsneTransform, y, len(our_json_files[0:10000]))
    elif tsne_comp == 3:
        scatter_3d(tsneTransform, y, len(our_json_files[0:10000]))

    assert False

if __name__ == "__main__":
    main()