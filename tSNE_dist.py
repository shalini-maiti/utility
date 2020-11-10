#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:12:37 2020

@author: shalini
"""
# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import json
import glob
#import tqdm
import cv2

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import imutils

# We import seaborn to make nice plots.
import seaborn as sns

# We'll generate an animation with matplotlib and moviepy.
#from moviepy.video.io.bindings import mplfig_to_npimage
#import moviepy.editor as mpy

#digits = load_digits()
#print(digits.data.shape) 

our_labels="/data3/datasets/mano_bg_like_24_with_depth_convertible_masks_24d/TRAIN/labels/"
freihands_labels = "/data3/datasets/FinalDataset/FreihandsTrainData/labels/"
shreyas_ds_labels = "/data3/datasets/mano_bg_hand_lighting_variation_without_rand_noise_wrap_27e/ho3d_varied/labels/"
stb_labels = "/data3/datasets/FinalDataset/STBRandomAll/labels/"
mhp_10k_labels = "/data3/datasets/FinalDataset/MultiviewHandPose10k/labels/"

#aug_labels="/data3/datasets/mano_bg_fixed_all_with_arm_25/TRAIN/labels_aug/"
our_image_folder = "/data3/datasets/mano_bg_like_24_with_depth_convertible_masks_24d/TRAIN/images/"
#print(len(images))

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

def prep_and_transform_img(our_images, freihands_images):
    all_images = []    
    im_stuff = []
    # Load image data
    for f in our_images:
        new_image = cv2.imread(f)
        all_images.append(np.ndarray.flatten(new_image))
        im_stuff.append([0])
    
    for f in freihands_images:
        new_image = cv2.imread(f)
        all_images.append(np.ndarray.flatten(new_image))
        im_stuff.append([1])
    #print(X.shape, y.shape)
    
    pass

def prep_and_transform(our_json_files, frei_json_files, sd_json_files, 
                       input_img, minScale, maxScale, minRot, maxRot, stb_json_files,
                       mhp_10k_json_files):
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
                all_label_data.append(scaled_2d[jointsMap][1:]) # [1: ] To forget the wrist joint
                js_stuff.append([0])
                y = y + 1
                print(y)
    
    for j_file in frei_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[1:])  # [1: ] To forget the wrist joint
            js_stuff.append([1])
            
        
    for j_file in sd_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[1:])  # [1: ] To forget the wrist joint
            js_stuff.append([2])
    
    for j_file in stb_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[1:])  # [1: ] To forget the wrist joint
            js_stuff.append([3])
            
    for j_file in mhp_10k_json_files:
        with open(j_file, 'r') as f:
            dat = json.load(f)
            pts2DHand = np.array(dat['hand_pts'], dtype='f')
            #all_target_data.append(jointsMap)
            all_label_data.append(pts2DHand[multiviewJointsMap][1:])  # [1: ] To forget the wrist joint
            js_stuff.append([4])
            
    all_label_data = np.array(all_label_data)
    all_label_data = all_label_data.reshape(all_label_data.shape[0], -1)

    print(all_label_data.shape) 

    X = all_label_data
    y =  np.array(js_stuff)
    #y = [0]*len(our_json_files) + [1]*len(frei_json_files)
    y = np.reshape(y, (all_label_data.shape[0], ))
    #y = np.concatenate(y, np.ones((len(freihands_images))))
    #print(X[0], y[0]) 
    print(X.shape, y.shape)
    tsneTransform = TSNE(n_components=2, random_state=RS).fit_transform(X)
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
    
    
    ax.scatter(x[colors ==0, 0], x[colors == 0, 1], c='r', marker='o', label="24d-28") # Ours
    ax.scatter(x[colors ==1 , 0], x[colors == 1, 1], c='b', marker='^', label="Freihands") #Frei
    ax.scatter(x[colors ==2 , 0], x[colors == 2, 1], c='g', marker='*', label="HO3D Sampled") #HO3D
    ax.scatter(x[colors ==3 , 0], x[colors == 3, 1], c='k', marker="4", label="STB") #STB
    ax.scatter(x[colors ==3 , 0], x[colors == 3, 1], c='c', marker='H', label="MultiviewHandPose") #STB
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
    
    ax.scatter(x[colors ==0 , 0], x[colors == 0, 1], x[colors ==0, 2], c='r', marker='o', label="1-6") # Ours
    ax.scatter(x[colors ==1 , 0], x[colors == 1, 1], x[colors ==1, 2], c='b', marker='^', label="Freihands") #Frei
    ax.scatter(x[colors ==2 , 0], x[colors == 2, 1], x[colors ==2, 2], c='g', marker='s', label="HO3D") #SD
    ax.scatter(x[colors ==3 , 0], x[colors == 3, 1], x[colors ==3, 2], c='k', marker="4", label="STB") #STB    
    #ax.scatter(x[0:len_our_json_files, 0], x[0:len_our_json_files, 1], x[0:len_our_json_files, 2],  c='r', marker='o')
    #ax.scatter(x[len_our_json_files:-1, 0], x[len_our_json_files:-1, 1], x[0:len_our_json_files, 2], c='b', marker='^')
    #ax.scatter(x[len_our_json_files:-1, 0], x[len_our_json_files:-1, 1], x[0:len_our_json_files, 2], c='g', marker='-')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
def main():
    minScale = 0.7
    maxScale = 1.4
    minRot= -180
    maxRot= 180
    our_json_files = [f for f in glob.glob(our_labels + "*.json")]
   
    our_json_files = random.sample(our_json_files, 10000)
    frei_json_files = [f for f in glob.glob(freihands_labels + "*.json")]
    ds_json_files = [f for f in glob.glob(shreyas_ds_labels + "*.json")]
    stb_json_files = [f for f in glob.glob(stb_labels + "*.json")]
    mhp_10k_json_files = [f for f in glob.glob(mhp_10k_labels + "*.json")] 
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
    img_src = [f for f in glob.glob(our_image_folder + "*.png")][0]
    print(img_src)
    input_img = cv2.imread(img_src)
    tsneTransform, y = prep_and_transform(our_json_files, frei_json_files, ds_json_files, 
                                          input_img, minScale, maxScale, minRot, maxRot, stb_json_files,
                                          mhp_10k_json_files)

    scatter_2d(tsneTransform, y, len(our_json_files[0:5000]))
    
    '''
    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(input_img, cmap=plt.cm.gray, interpolation='nearest')
    ax0.scatter(pts2DHand[:, 0], pts2DHand[:, 1], c='b', marker='o')
    ax1.imshow(rot_Img, cmap=plt.cm.gray, interpolation='nearest')
    ax1.scatter(rotPts2d[:, 0], rotPts2d[:, 1], c='b', marker='o')
    ax2.imshow(scaled_img, cmap=plt.cm.gray, interpolation='nearest')
    ax2.scatter(scaled_2d[:, 0], scaled_2d[:, 1], c='b', marker='o')
    plt.show()
    '''
    assert False

main()
#f, ax, sc, txts = scatter(digits_proj, y)


'''
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2))

    # We create a scatter plot.
    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    #ax = plt.add_subplot(111, projection='3d')
    #ax = Axes3D(fig)
    #ax.set_xlim(-500, 500)
    #ax.set_ylim(-500, 500)
    #ax.set_zlim(-500,500)
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(0, 500)
    plt.ylim(0,500)

    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    plt.show()
    
    plt.savefig('digits_tsne-generated.png', dpi=120)

    return fig, ax, sc, txts
'''