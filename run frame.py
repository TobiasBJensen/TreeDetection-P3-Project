# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import pickle
import cv2
from os import path
import os


# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('test3_color', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


frameSet = read_list()

num = 1
pause = True
for frame in frameSet:
    cv2.imshow("Color Stream", frame[0])
    print(frame[1])
    # esc exit, 's' start and 'p'
    while pause:
        key = cv2.waitKey(1)
        if key == ord('s'):
            pause = False

        if key == ord('c'):
            cv2.imwrite(os.path.join('test', f'tree{num}.jpg'), frame[0])
            pause = False
            num = num + 1

    key = cv2.waitKey(20)
    if key == ord('p'):
        pause = True

    if key == 27:
        cv2.destroyAllWindows()
        exit()