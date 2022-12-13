# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import pickle
import cv2
from os import path
import os

click_x, click_y = 0, 0


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_x, click_y
        click_x, click_y = x, y


# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('trainingBagFiles/test3', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


frameSet = read_list()

temp_color = cv2.imread("test/Test3.3.png")
temp_gray = cv2.cvtColor(temp_color, cv2.COLOR_BGR2GRAY)
test_mask = cv2.inRange(temp_color, (250, 50, 0), (255, 60, 0))

# test
#cv2.waitKey(0)
# test

#print(frameSet)
num = 1
pause = True
for frame in frameSet:
    cv2.imshow("Color Stream", frame[1])

    frame_gray = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
    outputTemplate = cv2.matchTemplate(frame_gray, temp_gray, cv2.TM_SQDIFF_NORMED)
    if outputTemplate <= 0.3:
        temp_gray = cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(frame[1], (0, 0, 254), (0, 0, 255))
        h, w = mask.shape[:2]
        mask_zeros = np.zeros((h + 2, w + 2), np.uint8)
        mask_inv = cv2.bitwise_not(mask)
        flood_mask = mask.copy()
        flood_test_mask = test_mask.copy()
        cv2.imshow("flood", mask)
        cv2.setMouseCallback("flood", click_event)
        cv2.waitKey(0)
        cv2.destroyWindow("flood")
        cv2.floodFill(flood_mask, mask_zeros, (click_x, click_y), 255)
        cv2.floodFill(flood_test_mask, mask_zeros, (0, 0), 255)
        flood_test_mask = cv2.bitwise_not(flood_test_mask)
        sub = cv2.bitwise_and(flood_test_mask, flood_test_mask, mask=mask_inv)
        contours, hierarchy = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        count = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(f"Nr: {count} have pixel area: {area}")
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                M['m00'] = 1

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(sub, f'Nr: {count}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            count = count + 1
        cv2.imshow("test", sub)
        print("")
        cv2.waitKey(0)
        cv2.destroyWindow("test")
        pause = True

    # esc exit, 's' start and 'p'
    while pause:
        key = cv2.waitKey(1)
        if key == ord('s'):
            pause = False

        if key == ord('c'):
            temp_gray = cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(frame[1], (0, 0, 254), (0, 0, 255))
            h, w = mask.shape[:2]
            mask_zeros = np.zeros((h + 2, w + 2), np.uint8)
            mask_inv = cv2.bitwise_not(mask)
            flood_mask = mask.copy()
            flood_test_mask = test_mask.copy()
            cv2.imshow("flood", mask)
            cv2.setMouseCallback("flood", click_event)
            cv2.waitKey(0)
            cv2.destroyWindow("flood")
            cv2.floodFill(flood_mask, mask_zeros, (click_x, click_y), 255)
            cv2.floodFill(flood_test_mask, mask_zeros, (0, 0), 255)
            flood_test_mask = cv2.bitwise_not(flood_test_mask)
            sub = cv2.bitwise_and(flood_test_mask, flood_test_mask, mask=mask_inv)
            contours, hierarchy = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
            count = 1
            for cnt in contours:
                area = cv2.contourArea(cnt)
                print(f"Nr: {count} have pixel area: {area}")
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    M['m00'] = 1

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(sub, f'Nr: {count}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                count = count + 1
            cv2.imshow("test", sub)
            print("")
            cv2.waitKey(0)
            cv2.destroyWindow("test")
            pause = False
            num = num + 1

    key = cv2.waitKey(200)
    if key == ord('p'):
        pause = True

    if key == 27:
        cv2.destroyAllWindows()
        exit()