import cv2
import numpy as np

# Make ROI
img = cv2.imread('RGB_color_pic_Color.png')



roi = img[0:720, 120:600] #[y-start : y-stop, x-start: x-stop]
# Color Thresholding for Trunk
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#Colorthreshold might be changed
minThresh = np.array([20,28,30])
maxThresh = np.array([114,100, 115])

mask = cv2.inRange(hsv, minThresh, maxThresh)
res = cv2.bitwise_and(roi, roi, mask=mask)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

closing = cv2.bitwise_and(roi, roi, mask=opening) #Color res, after opening/closing



cv2.imshow('output', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Make Binary

# Add morphology to reduce noise


# Make rectangle around found Trunk



#grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#retval, threshold = cv2.threshold(grayscaled, 50, 255, cv2.THRESH_BINARY)
#gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#cv2.imshow('Grayscaled Img', grayscaled)
#cv2.imshow('Binary Img', threshold)
