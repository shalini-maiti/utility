'''
Generate an .avi video from a folder of images.
'''

import cv2
import glob

img_array = []
img_dir = glob.glob('<Insert input image folder>/*.jpg')
video_destination = " " # Output .avi video address
print((img_dir[0].split('/')[8])[:-4])
img_dir = sorted(img_dir, key=lambda img: int(img.split('/')[8][:-7]))

for filename in img_dir:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    img_array.append(img)


out = cv2.VideoWriter(video_destination, cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

for i in range(len(img_array)):

    out.write(img_array[i])
out.release()