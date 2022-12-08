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
    with open('trainingBagFiles/listfile', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


frameSet = read_list()
test = cv2.imread("test/Test1.2.png")
test_mask = cv2.inRange(test, (250, 50, 0), (255, 60, 0))

temp_color = cv2.imread("test/tree1.PNG")
temp_gray = cv2.cvtColor(temp_color, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(temp_color, (0, 0, 254), (0, 0, 255))

cv2.imshow("test", test_mask)
cv2.waitKey(0)

#print(frameSet)
num = 1
pause = True
for frame in frameSet:
    cv2.imshow("Color Stream", frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outputTemplate = cv2.matchTemplate(frame_gray, temp_gray, cv2.TM_SQDIFF_NORMED)
    if outputTemplate <= 0.1:
        print("hit")
        #pic = cv2.bitwise_or(frame, frame, mask=mask_inv)
        frame[mask > 0] = (0, 0, 255)
        cv2.imshow("test", frame)
        pause = True

    # esc exit, 's' start and 'p'
    while pause:
        key = cv2.waitKey(1)
        if key == ord('s'):
            pause = False

        if key == ord('c'):
            cv2.imwrite(os.path.join('test', f'tree{num}.png'), frame)
            pause = False
            num = num + 1

    key = cv2.waitKey(180)
    if key == ord('p'):
        pause = True

    if key == 27:
        cv2.destroyAllWindows()
        exit()