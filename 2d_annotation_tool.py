import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Arrow, Circle
import json

#from matplotlib import mpl_connect, mpl_disconnect

folder_name = " " # Insert the address of the image folder
activateInteraction = False
#coords = {str(i): np.array([0, i]) for i in range(21)}
coords = np.ones((21, 2))
counter = 0
fig, ax = plt.subplots()
json_folder = "" # Insert the address where the labels should be saved
image_names = [img for img in glob.glob(folder_name+"*.png")]
#images = [cv2.imread(img) for img in image_names]
filecounter = 0

'''
def toggle_point_generation(fig):
  activateInteraction = not activateInteraction
  if activateInteraction:
    cid = fig.canvas.mpl_connect('button_press_event', onclick(e))
  else:
    cid = fig.canvas.mpl_disconnect('button_press_event', onclick(e))
'''

def onclick(e):
  global counter
  global filecounter
  global ax
  if counter < 21:
    #coords[str(counter)] = np.array([e.xdata, e.ydata])
    coords[counter] = np.array([e.xdata, e.ydata])
    #ax.add_patch(Circle((e.xdata, e.ydata), radius=10, color='green'))
    ax.scatter(e.xdata, e.ydata, label=str(counter))
    plt.show()
    plt.legend(loc="best")
    print(coords[counter])
    counter = counter + 1
  else:
    # Ask if Save to json, throw an alert and move to the next image
    if filecounter < len(image_names) - 1:
      filepath = image_names[filecounter].split("/")[8][:-4]
      save_to_json(coords, filepath, json_folder)
      filecounter = filecounter + 1
      set_next_img(filecounter)
      counter = 0
      print("counter", counter)
      print("filecounter", filecounter)
    pass

def save_to_json(pos_for_labelling, filepath, Labelfolder):
    label_dict = {}
    json_file = Labelfolder + filepath + '.json'

    print(json_file)
    #print(len(sp[1:]))
    #print(pos_for_labelling)
    print(pos_for_labelling.tolist())
    label_dict['hand_pts'] = pos_for_labelling.tolist()

    label_dict['is_left'] = 0
    g = open(json_file, 'w')
    json.dump(label_dict, g)
    pass

def set_next_img(filecounter):
  ax.clear()
  ax.imshow(cv2.imread(image_names[filecounter]))
  plt.show()


def main():
  ax.imshow(cv2.imread(image_names[filecounter]))

  # add a toggle interaction button
  #activate_int = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Activate')
  #activate_int.on_clicked(toggle_point_generation)
  cid = fig.canvas.mpl_connect('button_press_event', onclick)
main()