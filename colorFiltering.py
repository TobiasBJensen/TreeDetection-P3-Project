import cv2
import numpy as np
#This function colorThresholding extracts color in defined min and max BGR values.

#Variables for the function input
img = cv2.imread('RGB_color_pic_Color.png')
minThreshSky = np.array([250, 250, 250])# ([minH, minS, minV])
maxThreshSky = np.array([255, 255, 255])# ([maxH, maxS, maxV])
minThreshTree = np.array([])
maxThreshTree
def colorThresholding(roi, minT, maxT, kernel):
    #roi might be deleted
    #roi = img[0:100, 0:100] #[y-start : y-stop, x-start: x-stop]
    # Color Thresholding for Trunk
    hsv = img #Converted to hsv
    #cv2.imshow('HSV', hsv)
    mask = cv2.inRange(hsv, minT, maxT)
    res = cv2.bitwise_and(roi, roi, mask=mask) #If you want the result at Binary


    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    closing = cv2.bitwise_and(roi, roi, mask=closing) #Color res, after opening/closing
    #opening = cv2.bitwise_and(roi, roi, mask=opening)  # Color res, after opening/closing
    #cv2.imshow('opening', closing)
    #cv2.waitKey(0)
    cv2.imshow('Closing', closing)
    cv2.waitKey()
    return closing, opening, mask


def main():
    ClosingRGB, OpeningRGB, mask = colorThresholding(img, minThreshSky, maxThreshSky, kernel=np.ones((5, 5), np.uint8))

    while True:
      #  cv2.imshow('closing in Color', closing)

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
