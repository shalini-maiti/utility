'''

Read hand labels from a text file and save them to
a json file after inverting y and x coordinates.

'''
import numpy as np
import json

label_file = '/data3/datasets/mano_imagenet_vert_colour_2/TOTAL/labels.txt'
labels_folder = '/data3/datasets/mano_imagenet_vert_colour_2/TOTAL/labels/'

f = open(label_file, 'r')

for line in f:
    label_dict = {}
    sp = line.split()
    json_file = sp[0][0:-4] + '.json'
    print(json_file)
    #print(len(sp[1:]))
    label_dict['hand_pts'] = (np.array(sp[1:]).reshape(21,2))[:, [1, 0]].tolist() # Invert y, x to x, y

    label_dict['is_left'] = 0
    g = open(labels_folder + json_file, 'w')
    json.dump(label_dict, g)