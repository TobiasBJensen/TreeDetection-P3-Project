import os

import pyrealsense2 as rs
import numpy as np
import cv2
from sys import platform
from os import path
from colorFiltering import colorThresholding
from imutils.object_detection import non_max_suppression

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

        # Create opencv window to render image in
        #cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)

        for x in range(5):
            pipeline.wait_for_frames()

    except RuntimeError:
        print("Can't read the given file, are you sure it is the right type?")
        main()

    finally:
        return pipeline


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
    # depth threshold
    threshold = rs.threshold_filter(distance_min, distance_max)

    # runs the data through the filters
    frame = depth_to_disparity.process(depth_frame)
    frame = spatial.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)
    thres_frame = threshold.process(frame)

    # turn depth data into a numpy array
    depth_image = np.asanyarray(frame.get_data())
    # colorize the depth data and turn it into a numpy array
    colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

    # generates a binary image showing objects within a given depth threshold to isolate the trees
    depth_mask = cv2.inRange(depth_image, distance_min * 1000, distance_max * 1000)
    # runs closing algoritme on binary image
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    #cv2.imshow('hi', depth_mask)
    #findTrunk(depth_mask)

    # uses binary image as mask on color image, so it only shows the objects within the threshold
    masked = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

    return colorized_depth, masked, depth_mask, thres_frame


def cutTrunkAndGround(trunk, color_trunk_box):
    height, width = trunk.shape[:2]
    inputImg_threshold = cv2.inRange(trunk, (254, 0, 0), (255, 0, 0))

    contours, hierarchy = cv2.findContours(inputImg_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_coor = []
    for cnt in contours:
        x, y, cntWidth, cntHeight = cv2.boundingRect(cnt)
        cv2.rectangle(color_trunk_box, (x, y), (x + cntWidth, y + cntHeight), (120, 255, 0), 2)
        cv2.rectangle(trunk, (x, y), (x + cntWidth, y + cntHeight), (0, 0, 0), -1)
        box_coor.append((x + int(cntWidth/2), y + int(cntHeight/2)))

    if len(box_coor) > 0:
        ground = int(sum([item[1] for item in box_coor]) / len(box_coor))
    else:
        ground = height
    #print(ground)

    #print(box_coor)
    lineBetween = []
    #print(len(box_coor))
    box_coor.sort(reverse=True)
    #print(box_coor)
    while len(box_coor) > 1:
        first_coor = box_coor.pop(0)
        second_coor = box_coor[0]
        #print(first_coor, second_coor)
        lineBetween.append(int(second_coor[0] + (first_coor[0] - second_coor[0])/2))

    #print(lineBetween)
    for obj in lineBetween:
        cv2.rectangle(trunk, (obj-5, 0), (obj+5, height), (0, 0, 0), -1)

    cv2.rectangle(trunk, (0, ground), (width, height), (0, 0, 0), -1)
    #cv2.imshow("Output", outputTemplate)
    #cv2.imshow("ROI", ROI)

    trunk = cv2.cvtColor(trunk, cv2.COLOR_BGR2GRAY)

    return trunk, color_trunk_box


def findTrunk(binayimage):
    inputImg = cv2.cvtColor(binayimage, cv2.COLOR_GRAY2BGR)
    height, width = binayimage.shape
    ROI = binayimage[(height // 2)+20:height-80, 0:width]
    ROIh, ROIw = ROI.shape
    ROI = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
    numberOfThemplates = 0
    themplateList = list()
    for trunk in os.listdir("Trunks"):
        if os.path.isfile(os.path.join("Trunks", trunk)):
            numberOfThemplates += 1
            themplate = cv2.imread(f"Trunks\\Test{numberOfThemplates}.png")
            themplateList.append(themplate)

    boxes = list()

    cv2.imshow("ROI", ROI)

    for i in range(numberOfThemplates):
        H, W = themplateList[i].shape[:2]
        outputTemplate = cv2.matchTemplate(ROI, themplateList[i], cv2.TM_SQDIFF_NORMED)

        (y_points, x_points) = np.where(outputTemplate <= 0.28)


        outputTemplate = cv2.cvtColor(outputTemplate, cv2.COLOR_GRAY2BGR)

        #print(x_points)

        for (x, y) in zip(x_points, y_points):

            box = ((x, y, x + W, y + H))
            boxes.append(box)



    boxes = non_max_suppression(np.array(boxes), overlapThresh=0.5)

    inputImg_C = inputImg.copy()
    for (x1, y1, x2, y2) in boxes:


        cv2.rectangle(outputTemplate, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.rectangle(ROI, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imshow("f",ROI)
        cv2.waitKey(1)
        cv2.rectangle(inputImg_C, (x1 -10, y1 + height - 80 - ROIh), (x2 + 10, y2 + height - 80 - ROIh), (255, 0, 0), 3)

    return inputImg_C

def findGrass(binaryImage):
    height, width = binaryImage.shape

    binaryImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)


    cv2.imshow("binaryImageInput", binaryImage)

    template2 = np.full((1, width), 255, dtype=np.uint8)
    template2 = cv2.cvtColor(template2, cv2.COLOR_GRAY2BGR)
    H, W = template2.shape[:2]

    cv2.imshow("template2", template2)

    outputTemplate = cv2.matchTemplate(binaryImage, template2, cv2.TM_SQDIFF_NORMED)

    (y_points, x_points) = np.where(outputTemplate <= 0.1)

    boxes = []

    outputTemplate = cv2.cvtColor(outputTemplate, cv2.COLOR_GRAY2BGR)

    for (x, y) in zip(x_points, y_points):
        boxes.append((x, y, x + W, y + H))

    boxes = non_max_suppression(np.array(boxes), overlapThresh=0)
    #print(boxes)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(outputTemplate, (x1, y1), (x2, y2), (255, 0, 0), 3)

    slicegrass = 0

    (y_grass, x_grass) = np.where((outputTemplate == (255, 0, 0)).all(axis=2))

    for (x, y) in zip(x_grass, y_grass):
        slicegrass += 1

    noGrassImage = binaryImage[0: height - slicegrass, 0: width, :]

    #cv2.imshow("Output", outputTemplate)
    #cv2.imshow("nograss", noGrassImage)

    #cv2.waitKey(0)

def findContures(Closing_bgr, color_image, depth_frame):
    contours, hierarchy = cv2.findContours(Closing_bgr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    Closing_bgr_C = Closing_bgr.copy()
    color_image_C = color_image.copy()
    Closing_bgr_C = cv2.cvtColor(Closing_bgr_C, cv2.COLOR_GRAY2BGR)
    #print(contours)
    depth_image = np.asanyarray(depth_frame.get_data())
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            x, y, width, height = cv2.boundingRect(cnt)

            box_dist = depth_image[y:y + height, x:x + width]
            box_dist = box_dist.reshape((box_dist.shape[0] * box_dist.shape[1], 1))
            box_dist = box_dist[np.nonzero(box_dist)]

            if len(box_dist):
               dist = (sum(box_dist) / len(box_dist))/1000
            else:
                dist = 0

            cv2.rectangle(Closing_bgr_C, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(Closing_bgr_C, f'Width: {width} & Height: {height}', (x, y + height + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(Closing_bgr_C, f'Depth: {round(dist, 2)}m', (x, y + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.rectangle(color_image_C, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(color_image_C, f'Width: {width} & Height: {height}', (x, y + height + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(color_image_C, f'Depth: {round(dist, 2)}m', (x, y + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    return Closing_bgr_C, color_image_C


def main():
    # If you want to run the same file a lot just write the name of the file below and set bagFileRun to True
    bagFileRun = ("Training1.bag", True)

    # if you want to loop the script then using input, to run through different bag files. Set loopScript to True
    loopScript = True

    pipeline = initialize(bagFileRun)


    while True:
        depth_frame, colorized_depth, color_image = getFrames(pipeline)

        # process depth data and isolates objects within a given depth threshold
        modified_colorized_depth, color_removed_background, depth_masked, depth_image = \
            removeBackground(depth_frame, color_image, distance_max=4, distance_min=0.2) # distance is in meters

        minThresh = np.array([20, 28, 30])  # ([minH, minS, minV])
        maxThresh = np.array([114, 100, 115])  # ([maxH, maxS, maxV])
        #minThresh = np.array([10, 20, 25])  # ([minH, minS, minV])
        #maxThresh = np.array([150, 120, 140])  # ([maxH, maxS, maxV])


        Closing_bgr, Opening_bgr, mask = \
            colorThresholding(color_removed_background, minThresh, maxThresh, kernel=np.ones((3, 3), np.uint8))



        Closing_bgr2, Opening_bgr, mask = \
            colorThresholding(color_removed_background, minThresh, maxThresh, kernel=np.ones((5, 5), np.uint8))
        # Render image in opencv window
        #cv2.imshow("Depth Stream", colorized_depth)
        #cv2.imshow("Color Stream", color_removed_background)
        #cv2.imshow("Closing(7, 7)", Closing_bgr)
        #cv2.imshow("CLosing(5, 5)", mask)
        #cv2.imshow("d", depth_masked)

        #findGrass(depth_masked)

        trunk_box = findTrunk(depth_masked)
        cv2.imshow("Trunk", trunk_box)

        # | Tobias test
        # |
        # v

        # Uses trunk_box to cut trunk and ground
        treeCrown_box, color_trunk_box = cutTrunkAndGround(trunk_box, color_image)

        # Simple contures used for testing
        depth_masked_trunk_box, color_image_box = findContures(treeCrown_box, color_image, depth_image)
        cv2.imshow("trunk_box", color_trunk_box)
        cv2.imshow("test", color_image_box)
        #cv2.imshow("test2", depth_masked_trunk_box)
        #cv2.imshow("ColorImage", color_image)
        # if pressed escape exit program
        key = cv2.waitKey(1)




        if key == 27:
            cv2.destroyAllWindows()
            if loopScript and not bagFileRun[1]:
                main()
            break

if __name__ == "__main__":
    main()
