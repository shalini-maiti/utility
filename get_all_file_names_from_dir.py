import os
import sys
import glob
import shutil
import numpy as np

# Use json folder(labels ) to get the list from directory

file_list_imgs = '/data3/datasets/FinalDataset/MultiviewHandPoseAugmented10K/MultiviewHandPoseAug10k.txt'
img_folder = '/data3/datasets/FinalDataset/MultiviewHandPoseAugmented10K/labels/'

#img_folder = '/home/shalini/Downloads/trial/labels'
#file_list_imgs = '/home/shalini/Downloads/TESTRight_rand.txt'


f = open(file_list_imgs, 'a')

count = 0
for file in os.listdir(img_folder):
	if file.endswith('.json'):
		fname = file[0:-5]
		print(fname)
		count = count + 1
		f.write(fname+'\n')

print(count)
f.close()