#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:39:39 2020

@author: shalini
"""
import os
import glob
import json
import cv2

metrics_incremental_folder = "/data3/results/mano_imagenet/OldFinetuningLogs27/"
output_file = "/data3/results/mano_imagenet/OldFinetuningLogs27/AllMetrics.json"
output_txt_file = "/data3/results/mano_imagenet/OldFinetuningLogs27/AllMetrics.txt"
video_folder = "/data3/results/mano_imagenet/OldFinetuningLogs27/videos/"

def main():
    # Iterate through all the folders, get to the metrics file
    metric_files = [f for f in glob.glob(metrics_incremental_folder+ "logs_finetune*")]
    # store path as dict key and au2d, 2dkps  as values
    print(len(metric_files))
    metric_dict = {}
    file1 = open(output_txt_file,"w")#append mode 

    for m_file_folder in metric_files:
        temp = {}
        temp["path"] = m_file_folder.split("/")[-1]
        temp["model"] = m_file_folder.split("/")[-1][13:]

        m_all_ds = [f for f in glob.glob(m_file_folder + "/RESULTS/ManoHandsInference*")]
        for ds in m_all_ds:
            #print(ds)
            
            xxx = [f for f in glob.glob(ds+"/"+"*.txt")]
            if len(xxx) > 0:
                save_video(ds)
                m_file = xxx[0]
                print(m_file)
                with open(m_file, 'r') as f:
                    m_file_name = ds.split("/")[-1]
                    lines = f.readlines()
                    temp["au2d" + m_file_name[18:]] = lines[7].split(" ")[2]
                    temp["2dkps" + m_file_name[18:]] = lines[8].split(" ")[3][7:-1]
                    file1.write(temp["model"] +" "+ m_file_name[18:] + " " + temp["2dkps" + m_file_name[18:]] + "\n")
        metric_dict[temp["model"]] = temp
    g = open(output_file, 'w')
    json.dump(metric_dict, g)
    file1.close()
    
def save_video(path_to_ds):
    videoname = path_to_ds.split("/")[-1] + "_" + path_to_ds.split("/")[-3]
    img_dir = glob.glob(path_to_ds + "/KPS2DStick/*.jpg")
    #img_dir = sorted(path_to_ds, key=lambda img: int(img.split('/')[-1][:-4]))
    img_dir = sorted(img_dir, key=lambda img: img.split('/')[-1][:-4])
    img_array = []
    for filename in img_dir:    
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #print(filename)
        img_array.append(img)
    out = cv2.VideoWriter(video_folder + videoname+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
    for i in range(len(img_array)):
        print(i)
        out.write(img_array[i])
    out.release()
    
# In json, we save it in the mano format ['hand_pts']: {[(x1, y1), ..., (x21, y21)]}
def save_to_json(pos_for_labelling, plot_counter, imgDir):
    label_dict = {}
    json_file = str(plot_counter) + '_mano' + '.json'
    print(json_file)
    #print(len(sp[1:]))
    label_dict['hand_pts'] = pos_for_labelling.T[:,[1,0]].tolist()
    
    label_dict['is_left'] = 0
    g = open(imgDir + json_file, 'w')
    json.dump(label_dict, g)
    pass

if __name__ == "__main__":
    main()
