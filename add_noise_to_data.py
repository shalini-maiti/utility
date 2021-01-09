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

input_img_folder = " " # Address of the input image folder
output_img_folder = " " # Address of the output image folder


def add_noise_to_image(img, noise_type, amount=0.3, mean=0.5, var=0.5):
    if noise_type == 's&p':
      # Add s&p noise to the image.
      noise_img = random_noise(img, noise_type, seed=42, clip=True, amount=amount )
    elif noise_type == 'gaussian':
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
    counter = len(img_files)
    for img_name in img_names_:
        img_src = input_img_folder + img_name + ".png"
        input_img = cv2.imread(img_src)
        img_dest = output_img_folder + img_name + ".png"

        # Gaussian
        #noisy_img = add_noise_to_image(input_img, 'gaussian', mean=0.0, var=0.005)

        # S&p
        noisy_img = add_noise_to_image(input_img, 's&p', amount=0.0005)


        cv2.imwrite(img_dest, noisy_img) #final_img

        print("Fin.", counter)
        counter = counter -1
        #assert False
    pass

main()