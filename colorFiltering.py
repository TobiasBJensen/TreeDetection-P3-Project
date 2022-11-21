import cv2
import numpy as np

# Make ROI

#Colorthreshold might be changed
minThresh = np.array([20,28,30])
maxThresh = np.array([114,100, 115])

def colorThresholding(img, minT, MaxT):
    roi = img[0:720, 120:600] #[y-start : y-stop, x-start: x-stop]
    # Color Thresholding for Trunk
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, minThresh, maxThresh)
    res = cv2.bitwise_and(roi, roi, mask=mask)

    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    closing = cv2.bitwise_and(roi, roi, mask=opening) #Color res, after opening/closing
    return closing

def main():
    img = cv2.imread('RGB_color_pic_Color.png')

    while True:
        colorThresholding(img, minThresh, maxThresh)

    cv2.imshow('closing in Color', closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Make Binary

# Add morphology to reduce noise


# Make rectangle around found Trunk



