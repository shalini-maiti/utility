import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('/data3/results/mano_imagenet/logs_finetune27b27c27_27e27d/RESULTS/ManoHandsInference_STBCountingAll/KPS2DStick/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('/data3/results/mano_imagenet/logs_finetune27b27c27_27e27d/RESULTS/ManoHandsInference_STBCountingAll/pred_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
# ALso sync the loss being drawn on a graph and merge it with the video 

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()