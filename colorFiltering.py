import cv2
import numpy as np
# This function colorThresholding extracts color in defined min and max BGR values.

# Used for testing
img = cv2.imread('RGB_color_pic_Color.png')

# Threshold used to remove the sky
minThresh = np.array([230, 230, 230])  # ([minB, minG, minR])
maxThresh = np.array([255, 255, 255])  # ([maxB, maxG, maxR])


# This function is used in the main program
def colorThresholding(image, minT, maxT, kernel):
    # roi might be deleted
    # roi = img[0:720, 120:600] #[y-start : y-stop, x-start: x-stop]
    # Color Thresholding for Trunk
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Converted to hsv

    mask = cv2.inRange(image, minT, maxT)
    res = cv2.bitwise_and(image, image, mask=mask)  # If you want the result at Binary

    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(image, image, mask=opening)  # Color res, after opening/closing

    final = cv2.subtract(image, result)
    binary = cv2.bitwise_not(opening)
    return final, binary


def main():
    finalImg, opening = colorThresholding(img, minThresh, maxThresh, kernel=np.ones((5, 5), np.uint8))

    while True:
        cv2.imshow('Skysubtract + closing + open', finalImg)
        cv2.imshow('test', opening)

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
