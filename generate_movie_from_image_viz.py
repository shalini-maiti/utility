import cv2
import numpy as np
import glob
 
img_array = []
img_dir = glob.glob('/data3/results/mano_imagenet/logs_finetune24e24f24df/RESULTS/ManoHandsInference_STBCountingAll/KPS2DStick/*.jpg')
print((img_dir[0].split('/')[8])[:-4])
img_dir = sorted(img_dir, key=lambda img: int(img.split('/')[8][:-7]))
    
for filename in img_dir:    
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    img_array.append(img)
 
 
out = cv2.VideoWriter('/data3/results/mano_imagenet/videos/ManoHandsInference_STBCountingAll_logs_finetune24e24f24df.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
# ALso sync the loss being drawn on a graph and merge it with the video 

for i in range(len(img_array)):
    
    out.write(img_array[i])
out.release()