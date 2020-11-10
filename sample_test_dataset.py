import os
import sys
import glob
import shutil
import numpy as np
import random

json_files_path = "/data3/datasets/multiview_hand_pose_dataset_v2/MultiviewHandPoseAugmentedAll/labels/"
image_files_path = "/data3/datasets/multiview_hand_pose_dataset_v2/MultiviewHandPoseAugmentedAll/images/"


output_test_img_path = "/data3/datasets/multiview_hand_pose_dataset_v2/MultiviewHandPoseAugmented10K/images/"
output_test_json_path = "/data3/datasets/multiview_hand_pose_dataset_v2/MultiviewHandPoseAugmented10K/labels/"

#output_train_img_path = "/data3/datasets/mano_arm_imagenet_vert_colour_and_bg_10_b_complete/TRAIN/images/"
#output_train_json_path = "/data3/datasets/mano_arm_imagenet_vert_colour_and_bg_10_b_complete/TRAIN/labels/"

total_number = 60000
train_number = 50000
#total_number = 62000
#train_number = 62000
#total_number = 93000
#train_number = 93000

test_number = total_number - train_number

json_all = [f for f in glob.glob(json_files_path + "*.json")]
#json_files = random.sample(json_all, total_number)


#json_train = json_files[0: train_number ]
#json_test = json_files[train_number : total_number ] # For datasets
json_test = random.sample(json_all, test_number) #Random sampling of STB dataset

#img_names_train = [f.split("/")[6][:-5] for f in json_train]
img_names_test = [f.split("/")[6][:-5] for f in json_test] # 6 for datasets, 5 for STB

#print(len(img_names_train))
print(len(img_names_test))

#assert False
'''
for r in img_names_train:
	print(r)
'''

#img_files = [f.split("/")[6][:-4] for f in glob.glob(image_files_path + "*.jpg")] # 6 for datasets, 5 for STB
 

#print(np.array(img_files))

# TRAIN
'''
for img_name in img_names_train:
    #print(image_files_path + img_name + ".png")
    src1 = image_files_path + img_name + ".png"
    dest1 = output_train_img_path
    #print(json_files_path + img_name + ".json")
    src2 = json_files_path + img_name + ".json"
    dest2 = output_train_json_path
    print("train", img_name)
    #assert False
    if os.path.exists(src1):
        #shutil.move(src1, dest1)
        #shutil.move(src2, dest1)
        shutil.copy(src1, dest1)
        shutil.copy(src2, dest2)
'''

# TEST
for img_name in img_names_test:
    #print(image_files_path + img_name + ".png")
    src1 = image_files_path + img_name + ".jpg"
    dest1 = output_test_img_path
    #print(json_files_path + img_name + ".json")
    src2 = json_files_path + img_name + ".json"
    dest2 = output_test_json_path
    print("test", img_name)
    #assert False
    if os.path.exists(src1):
        #shutil.move(src1, dest1)
        #shutil.move(src2, dest1)
        shutil.copy(src1, dest1)
        shutil.copy(src2, dest2)
    