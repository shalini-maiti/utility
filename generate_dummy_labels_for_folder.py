import os
import glob
import numpy as np
import json

img_folder = '/data3/datasets/FinalDataset/ho3d_qualitative/images/'
labels_folder = '/data3/datasets/FinalDataset/labels_ho/'



    
def save_to_json(pos_for_labelling, label_name, labelDir):
    label_dict = {}
    json_file = str(label_name) + '.json'
    print(json_file)
    label_dict['hand_pts'] = pos_for_labelling.tolist()
    
    label_dict['is_left'] = 0
    g = open(labelDir + json_file, 'w')
    json.dump(label_dict, g)
    pass
  

def main():
    json_all = [f for f in glob.glob(img_folder + "*.png")]
    print('size',len(json_all))
    json_files = json_all
    img_names_ = [f.split("/")[6][:-4] for f in json_files]
    
    
    for img_name in img_names_:
        pos_for_labelling = np.random.randint(480, size=(21,2))
        pos_for_labelling[0] = [0, 200]
        pos_for_labelling[1] = [480, 200]
        pos_for_labelling[2] = [480, 640]
        pos_for_labelling[3] = [0, 640]
        save_to_json(pos_for_labelling, img_name, labels_folder)
         
main()
