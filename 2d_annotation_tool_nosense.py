#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:20:52 2020

@author: shalini
"""

from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#fig = plt.figure()


coords = []

'''
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix, iy)

    global coords
    coords.append((ix, iy))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print(cid)
    if len(coords) == 42:
        fig.canvas.mpl_disconnect(cid)

    return coords
'''

root = Tk()
root.title('Manual 2D Hand Joint Annotation Tool')
#root.iconbitmap( "/data3/saved_results/Fails/10mar_600k_rot_online_aug/3048_mano.jpg")
folder_name = "/data3/saved_results/Fails/10mar_600k_rot_online_aug/"
# todo: Write a browse button to select folder via interface later

first_img = "/data3/saved_results/Fails/10mar_600k_rot_online_aug/3048_mano.jpg"
temp = [img_file for img_file in glob.glob(folder_name+"*.jpg")]

image_list = [ImageTk.PhotoImage(Image.open(img_file)) for img_file in glob.glob(folder_name+"*.jpg")]
#image_list = [cv2.imread(folder_name+'/'+img_file) for img_file in os.listdir(folder_name)]
#image_list = [Image.open(folder_name+'/'+img_file) for img_file in os.listdir(folder_name)]

#a = fig.add_subplot(111)
#a.imshow(cv2.imread(temp[0]))

#my_label = Label(image=ImageTk.PhotoImage(Image.open(first_img)))
#my_label = Label(image=ImageTk.PhotoImage(image_list[0]))
#my_label.grid(row=0, column=0, columnspan=3)

canvas = Canvas(root, height=480, width=640)
#canvas.pack(expand=YES)
canvas.image = image=ImageTk.PhotoImage(Image.open(first_img))
canvas.create_image(0,0, image = canvas.image, anchor="nw")


def forward(image_number):
    global my_label
    global button_forward
    global button_back
    canvas.delete("all")
    
    canvas = Canvas(root, height=480, width=640)
    #canvas.pack(expand=YES)
    
    canvas.image = image_list[image_number -1]
    canvas.create_image(0,0, image = canvas.image, anchor="nw")
    
    button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
    button_back = Button(root, text="<<", command=lambda: back(image_number-1))

    if image_number == 5:
        button_forward = Button(root, text=">>", state=DISABLED)

    #my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)
    button_forward.grid(row=1, column=2)

def back(image_number):
    global my_label
    global button_forward
    global button_back

    canvas.delete("all")
    canvas = Canvas(root, height=480, width=640)
    #canvas.pack(expand=YES)
    canvas.image = image_list[image_number -1]
    canvas.create_image(0,0, image = canvas.image, anchor="nw")

    button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
    button_back = Button(root, text="<<", command=lambda: back(image_number-1))

    if image_number == 1:
        button_back = Button(root, text="<<", state=DISABLED)

    #my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)
    button_forward.grid(row=1, column=2)



button_back = Button(root, text="<<", command=back, state=DISABLED)
button_exit = Button(root, text="Exit Program", command=root.quit)
button_forward = Button(root, text=">>", command=lambda: forward(2))


button_back.grid(row=1, column=0)
button_exit.grid(row=1, column=1)
button_forward.grid(row=1, column=2)

root.mainloop()