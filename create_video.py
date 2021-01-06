__author__ = 'mahdi'


import numpy as np
import cv2
import glob


def create_video_open(filename="output", fps=15.0, width=1280, height=800):

    # filename += ".avi"
    filename += ".mp4"

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #fourcc = cv2.VideoWriter(['output', cv2.CV_FOURCC('M', 'J', 'P', 'G'), 25, (width, height), True])
    #out = cv2.VideoWriter(filename, -1, fps, (width, height))
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    return out


def create_video_add_frame(out, frame):
    if not out is None:
        out.write(frame)


def create_video_close(out):
    if not out is None:
        out.release()


if __name__ == '__main__':
    print ("Creating video ...")
    img_dir = glob.glob('/data3/results/mano_imagenet/logs_27hz/RESULTS/ManoHandsInference_ho3d_qualitative/KPS2DStick/*.jpg')


    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #fourcc = cv2.VideoWriter(['output', cv2.CV_FOURCC('M','J','P','G'), 25, (640,480),True])
    out = cv2.VideoWriter('/data3/results/mano_imagenet/logs_27hz/RESULTS/ManoHandsInference_ho3d_qualitative/video_mixed_individual_withSLds.avi', fourcc, 3.0, (300, 300))
    img_filenames = sorted(img_dir, key=lambda img: int(img.split('/')[-1][:-7]))

    for filename in img_filenames:
        #filename = '/home/mahdi/TUG/3DTracking/box_keyframes/ATLASBoxVideosCANON_EOS/video1/frame0000{0}.jpeg'.format(i)
        orig_img = cv2.imread(filename)


        minisize = (300, 300)
        #orig_img = cv2.resize(orig_img, minisize)

        # write the flipped frame

        out.write(orig_img)

        cv2.imshow('frame',orig_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
