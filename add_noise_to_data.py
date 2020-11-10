#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:59:41 2020

@author: shalini
"""
import cv2
import numpy as np
from skimage.util import random_noise
import glob

input_img_folder = "/data3/datasets/mano_bg_hand_lighting_variation_without_bg_27b/TRAIN/images/"
output_img_folder = "/data3/datasets/mano_bg_hand_lighting_variation_without_rand_noise_wrap_27e/TRAIN/images/"

def add_noise_to_image(img, noise_type, amount=0.3, mean=0.5, var=0.5):
    # Add s&p noise to the image.
    #noise_img = random_noise(img, noise_type, seed=42, clip=True, amount=amount )
    
    # Add gaussian noise
    noise_img = random_noise(img, noise_type, seed=42, clip=True, mean=mean, var=var )
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
     
    # Display the noise image
    #cv2.imshow('noise', noise_img)
    #cv2.waitKey(0)
    return noise_img
    
def main():
    img_files = [f for f in glob.glob(input_img_folder + "*.png")]
    img_names_ = [f.split("/")[6][:-4] for f in img_files]
    print(img_names_[0])
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".png"
        #img_dest_eg = output_img_folder + img_name + "_eq"+ ".png"
        #img_dest_adeg = output_img_folder + img_name + "_adeq"+ ".png"
        


        #for row in range(pts2DHand.shape[0]):
            #cv2.circle(resized_img, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
            #cv2.circle(resized_mask, (pts2DHand[row, 0], pts2DHand[row, 1]), 5, (255, 255, 0), thickness=1, lineType=8, shift=0)
        noisy_img = add_noise_to_image(input_img, 'gaussian', mean=0.0, var=0.005)
        #noisy_img = add_noise_to_image(input_img, 's&p', amount=0.0005)
        
        #final_img = seg_using_gt_mask(input_img, input_mask)

        #cv2.imwrite(img_dest, final_img) #final_img
        #cv2.imwrite(img_dest_eg, img_eq) #final_img
        
        cv2.imwrite(img_dest, noisy_img) #final_img
        
        #cv2.imwrite(mask_dest, resized_mask)
        #cv2.imwrite(output_img_folder+"final.png", resized_img)
        print("Fin.", img_name)
        #assert False
    pass

main()