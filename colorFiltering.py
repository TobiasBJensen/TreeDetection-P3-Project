import cv2
import numpy as np
#This function colorThresholding extracts color in defined min and max BGR values.

#Variables for the function input
img = cv2.imread('RGB_color_pic_Color.png')
minThresh = np.array([20, 28, 30])# ([minH, minS, minV])
maxThresh = np.array([114, 100, 115])# ([maxH, maxS, maxV])
def colorThresholding(roi, minT, maxT, kernel):
    #roi might be deleted
    #roi = img[0:720, 120:600] #[y-start : y-stop, x-start: x-stop]
    # Color Thresholding for Trunk
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #Converted to hsv

    mask = cv2.inRange(hsv, minT, maxT)
    res = cv2.bitwise_and(roi, roi, mask=mask) #If you want the result at Binary


    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    #closing = cv2.bitwise_and(roi, roi, mask=closing) #Color res, after opening/closing
    #opening = cv2.bitwise_and(roi, roi, mask=opening)  # Color res, after opening/closing
    #cv2.imshow('opening', closing)
    #cv2.waitKey(0)
    return closing, opening, mask

def main():
    ClosingRGB, OpeningRGB, mask = colorThresholding(img, minThresh, maxThresh, kernel=np.ones((5, 5), np.uint8))

    while True:
        cv2.imshow('closing in Color', ClosingRGB)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break



    # if pressed escape exit program

    # Acces each frame from depth image
    # for f, frame in enumerate(depth_color_image):
    # print("Depth" + str(depth_frame.get_frame_number()))  # Acces frame number (Hvis det kan bruges til noget)
    # Acces each frame from Color image
    # for f, frame in enumerate(color_image):
    # Acces frame number
    # <print("Color" + str(color_frame.get_frame_number()))

    # if pressed escape exit program

if __name__ == "__main__":
    main()
