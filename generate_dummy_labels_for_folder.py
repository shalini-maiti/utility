import os
import glob
import numpy as np
import json

img_folder = '/data3/Segmentation/images/ho3d_qualitative/images/'
labels_folder = '/data3/Segmentation/images/ho3d_qualitative/labels/'



    
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
        pos_for_labelling = np.random.randint(460, size=(21,2))
        save_to_json(pos_for_labelling, img_name, labels_folder)
         
main()
