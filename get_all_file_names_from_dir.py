import os
import sys
import glob
import shutil
import numpy as np

# Use json folder(labels ) to get the list from directory

file_list_imgs = '/data3/datasets/mano_like_24d_more_images_with_shape_base_24ne/TRAINRight.txt'
labels_folder = '/data3/datasets/mano_like_24d_more_images_with_shape_base_24ne/TRAIN/labels/'

#img_folder = '/home/shalini/Downloads/trial/labels'
#file_list_imgs = '/home/shalini/Downloads/TESTRight_rand.txt'


f = open(file_list_imgs, 'a')

count = 0
labels_dir = sorted([(file[:-5]) for file in os.listdir(labels_folder) if 'json' in file])


for file in labels_dir:
	fname = str(file)
	print(fname)
	count = count + 1
	f.write(fname+'\n')

print(count)
f.close()