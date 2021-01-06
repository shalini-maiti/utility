#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:42:09 2020

@author: shalini
"""
import os
import shutil
import glob

input_folder = "/data3/results/mano_imagenet/logs_27hz/"
output_folder = "/data3/results/mano_imagenet/logs_27hz/RESULTS/"

def add_new_checkpoints_to_folder(input_folder, output_folder, new_chkpt, old_chkpt):
    #Purge current checkpoints and checkpoint file
    #model_checkpoint_path: "/data/maiti/data3/datasets/mano_24d_with_all_variations_24z/RESULTS/logs/model.ckpt-132195"
    checkpoint_filename = os.path.join(input_folder, "RESULTS", "checkpoints", "checkpoint")
    with open(checkpoint_filename, 'w') as f:
        f.write("model_checkpoint_path: "+'"'+input_folder+"RESULTS/checkpoints/model.ckpt-"+ str(new_chkpt)+'"') 
    f.close()
    

    file_list_old = [input_folder+"logs/model.ckpt-"+old_chkpt+".data-00000-of-00001", 
                     input_folder+"logs/model.ckpt-"+old_chkpt+".index",
                     input_folder+"logs/model.ckpt-"+old_chkpt+".meta"]
    
    file_list_new = [input_folder+"logs/model.ckpt-"+new_chkpt+".data-00000-of-00001", 
                     input_folder+"logs/model.ckpt-"+new_chkpt+".index",
                     input_folder+"logs/model.ckpt-"+new_chkpt+".meta"]
    
    '''       
    for j in file_list_old:
        if not os.path.exists(os.path.join(input_folder, 'logs', j)):
            shutil.move(j, os.path.join(input_folder, 'logs'))
        
    for i in file_list_new:
        if not os.path.exists(os.path.join(output_folder, 'checkpoints', i)):
            shutil.copy(i, os.path.join(output_folder, 'checkpoints'))
    '''
    
def main():
    list_of_models = [f.split("/")[-1][11:] for f in glob.glob(input_folder+"logs/" +"*.meta")]
    more_list_of_models = sorted([int(f[:-5]) for f in list_of_models])
    files_list = [str(f) for f in more_list_of_models]
    
    for index, model in enumerate(files_list):
        counter = 0
        print(model)
        if counter>1 and counter<len(more_list_of_models):
            add_new_checkpoints_to_folder(input_folder, output_folder, model, model)
            counter = counter + 1
            
            #assert False
        else:
            add_new_checkpoints_to_folder(input_folder, output_folder, model, model)
            counter = counter + 1
            #assert False
        ds = os.path.join(output_folder, "ManoHandsInference_validationSet")
        txtfileIn = "Metrics.txt"
        txtfileOut = "AllMetrics.txt"
        fx = os.system('python /data3/Training/handkpreg-master/deeplab/inference.py')
        print("FX", fx)
        if fx == 0:
            save_metrics_to_file(ds, txtfileIn, txtfileOut, model)
        
        

    
def save_metrics_to_file(ds, txtfileIn, txtfileOut, modelname):
        temp = {}
        print(txtfileIn)
        with open(os.path.join(ds, txtfileIn), 'r') as f:
            m_file_name = ds.split("/")[-1]
            lines = f.readlines()
            temp["au2d" + m_file_name[18:]] = lines[7].split(" ")[2]
            temp["2dkps" + m_file_name[18:]] = lines[8].split(" ")[3][7:-1]
            fileOut = open(os.path.join(ds, txtfileOut),"a")
            fileOut.write(modelname +" "+ m_file_name[18:] + " " + temp["2dkps" + m_file_name[18:]] + "\n")    
           
        
if __name__ == "__main__":
    main()
    
    
def save_to_file(pos_array, filename, image_name):
    with open(filename, 'a') as f:
        f.write(image_name) 
        f.write(' ')
        for col in range(pos_array.shape[1]):
            f.write(' ')
            #print(pos_array[0, col])
            f.write(str(pos_array[0, col]))
            f.write(' ')
            f.write(str(pos_array[1, col]))
        f.write('\n')                 
    pass