'''
Convert json labels to text file
'''
import os
import glob
import numpy as np
import json

label_file = ' ' # Output text file
labels_folder = ' ' # Input json labels file

f = open(label_file, 'r')

for line in f:
    label_dict = {}
    sp = line.split()
    json_file = sp[0][0:-4] + '.json'
    print(json_file)
    #print(len(sp[1:]))
    label_dict['hand_pts'] = (np.array(sp[1:]).reshape(21,2)).tolist()

    label_dict['is_left'] = 0
    g = open(labels_folder + json_file, 'w')
    json.dump(label_dict, g)