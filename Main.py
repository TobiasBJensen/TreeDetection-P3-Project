import pyrealsense2 as rs
import numpy as np
import cv2
from sys import platform
from os import path
from colorFiltering import colorThresholding

def pathToFile(bagFileRun):
    if not bagFileRun[1]:
        # write command or name of the bag file you want to run
        bagFile = input("Input Bag file name or type \"exit\" to end script: ")
    else:
        bagFile = bagFileRun[0]

    # command that exits the script
    if bagFile == "exit":
        exit()

    # runs this part for Windows systems
    if platform == "win32":
        # looks in local folder
        pathToBag = f"trainingBagFiles\\{bagFile}"
        # if the file is not in the local folder, then it looks in the external hard drives folder
        if not path.isfile(pathToBag):
            pathToBag = f"D:\\Rob3_Gruppe_6_Realsense_data\\BagfileTest\\{bagFile}"

    # runs this part for Mac systems
    if platform == "darwin":
        # looks in local folder
        pathToBag = f"trainingBagFiles/{bagFile}"
        # if the file is not in the local folder, then it looks in the external hard drives folder
        if not path.isfile(pathToBag):
            pathToBag = f"D:/Rob3_Gruppe_6_Realsense_data/BagfileTest/{bagFile}"

    # if the name given correspond to a file in set folders return the path, else try again
    if path.isfile(pathToBag):
        return pathToBag
    else:
        print("Can't find a file with that name")
        if bagFileRun[1]:
            exit()
        main()

def initialize(bagFileRun):
    # Path towards a bag file
    pathToRosBag = pathToFile(bagFileRun)

    try:
        # Create pipeline
        #print(pathToRosBag)
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, pathToRosBag)

        # Configure the pipeline to stream both the depth and color streams
        # Streams must be setup the same way they were recorded
        # You can use RealSense viewer to figure out what streams, and their corresponding formats and FPS, are available in a bag file
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

        # Start streaming from file
        pipeline.start(config)

        for x in range(5):
            pipeline.wait_for_frames()

    except RuntimeError:
        print("Can't read the given file, are you sure it is the right type?")
        main()

    finally:
        return pipeline


def blobDetection(image):
    height, width = image.shape
    img = image[0:height - 150, 0:width]
    params = cv2.SimpleBlobDetector_Params()
    #params.minThreshold = 0
    #params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    #params.minDistBetweenBlobs = 50
    params.filterByArea = False
    params.minArea = 200
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    imageWithKeypoints = cv2.drawKeypoints(img, keypoints, img)
    #imageWithKeypoints = cv2.drawKeypoints(image, keypoints, np.zeros((1,1)), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("key", imageWithKeypoints)


    print("blobs:", len(keypoints))


def getFrames(pipeline):
    # Create colorizer object for the depth stream
    colorizer = rs.colorizer()

    # Align RGB to depth
    alignD = rs.align(rs.stream.depth)
    # Align depth to RGB
    alignC = rs.align(rs.stream.color)

    # Get frames
    frameset = pipeline.wait_for_frames()
    frameset = alignD.process(frameset)
    depth_frame = frameset.get_depth_frame()
    color_frame = frameset.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    return depth_frame, colorized_depth, color_image


def removeBackground(depth_frame, color_image, distance_max, distance_min):
    # config for the different filters
    # filter for colorizing depth data
    colorizer = rs.colorizer(0)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    # fill holes by giving unknown pixels the value of the neighboring pixel closest to the sensor
    hole_filling = rs.hole_filling_filter(2)
    # defines boarders and smoothen the depth data
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.3)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 5)

    # runs the data through the filters
    frame = depth_to_disparity.process(depth_frame)
    frame = spatial.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

    # turn depth data into a numpy array
    depth_image = np.asanyarray(frame.get_data())
    # colorize the depth data and turn it into a numpy array
    colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

    # generates a binary image showing objects within a given depth threshold to isolate the trees
    depth_mask = cv2.inRange(depth_image, distance_min * 1000, distance_max * 1000)
    # runs closing algoritme on binary image
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # uses binary image as mask on color image, so it only shows the objects within the threshold
    masked = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

    return colorized_depth, masked, depth_mask


def findContures(Closing_bgr, color_image):
    contours, hierarchy = cv2.findContours(Closing_bgr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    Closing_bgr_C = Closing_bgr.copy()
    color_image_C = color_image.copy()
    Closing_bgr_C = cv2.cvtColor(Closing_bgr_C, cv2.COLOR_GRAY2BGR)
    #print(contours)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, width, height = cv2.boundingRect(cnt)

            cv2.rectangle(Closing_bgr_C, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.rectangle(color_image_C, (x, y), (x + width, y + height), (0, 0, 255), 2)

    return Closing_bgr_C, color_image_C


def imageShow(bagFileRun, color_image, binary_image, depth_binary):
    cv2.imshow("Binary Box", binary_image)
    cv2.imshow("Color Box", color_image)
    cv2.imshow("Depth Binary", depth_binary)


    # if pressed escape exit program
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        if bagFileRun[2] and not bagFileRun[1]:
            main()
        exit()


def simplegrass(image):

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 5000
    params.maxArea = 1000000
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False


def findCanopy(image):
    height, width = image.shape
    img = image[0:height - 150, 0:width]

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    cv2.imshow("edge", img)


def main():
    # If you want to run the same file a lot, then set the second argument in bagFileRun to True
    # Write the name of the file you want to run in the first argument in bagFileRun.
    # if you want to loop the script then using input, to run through different bag files. Set last argument to True
    bagFileRun = ("Training2.bag", True, False)

    # This function initializes the pipline
    pipeline = initialize(bagFileRun)

    while True:
        # This function pulls the frames from the pipeline
        depth_frame, colorized_depth, color_image = getFrames(pipeline)

        # Process depth data and isolates objects within a given depth threshold
        modified_colorized_depth, color_removed_background, depth_binary = \
            removeBackground(depth_frame, color_image, distance_max=4, distance_min=0.2) # distance is in meters

        # Process color data and isolates objects within a given color threshold
        minThresh = np.array([20, 28, 30])  # ([minH, minS, minV])
        maxThresh = np.array([114, 100, 115])  # ([maxH, maxS, maxV])
        Closing_bgr, Opening_bgr, mask = \
            colorThresholding(color_removed_background, minThresh, maxThresh, kernel=np.ones((7, 7), np.uint8))

        # Finds contures and sets bounding boxes around the trees
        Closing_bgr_box, color_image_box = findContures(Closing_bgr, color_image)

        # Render images in opencv window
        imageShow(bagFileRun, Closing_bgr_box, color_image_box, depth_binary)


if __name__ == "__main__":
    main()
