import os
from os import path
from sys import platform
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from imutils.object_detection import non_max_suppression
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
        # print(pathToRosBag)
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, pathToRosBag)

        # Configure the pipeline to stream both the depth and color streams
        # must be setup the same way they were recorded
        # You can use RealSense viewer to figure out what streams, and their corresponding formats and
        # FPS, are available in a bag file
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

        # Start streaming from file
        pipeline.start(config)

        # Create opencv window to render image in
        cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)

        # Skips the first 5 frames, because the camera uses them to calibrate
        for x in range(5):
            frame = pipeline.wait_for_frames()
            # saves the number of a frame to know when it has looped
            if x == 4:
                frame_number = frame.get_frame_number()

    # If the pipeline can't read the file, it will start at main again
    except RuntimeError:
        print("Can't read the given file, are you sure it is the right type?")
        main()

    finally:
        return pipeline, frame_number


def getFrames(pipeline, frame_number_start):
    # Create colorizer object for the depth stream
    colorizer = rs.colorizer()

    # Align RGB to depth
    alignD = rs.align(rs.stream.depth)
    # Align depth to RGB
    # alignC = rs.align(rs.stream.color)

    # Get frames
    frame_set = pipeline.wait_for_frames()
    frame_set = alignD.process(frame_set)
    depth_frame = frame_set.get_depth_frame()
    color_frame = frame_set.get_color_frame()
    # Get frame number and check if the video has looped
    frameNumber = frame_set.get_frame_number()
    if frameNumber <= frame_number_start:
        videoDone = True
    else:
        videoDone = False

    # Gets camera focal length and principal point of image for depth frame
    depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

    # Converts color frame to np array and switches from RGB to BGR
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # Displays depth data as a colorized image and converts it to np array
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    return depth_frame, colorized_depth, color_image, videoDone, depth_intrinsics


def removeBackground(depth_frame, color_image, sky_binary, distance_max, distance_min):
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
    thresh_frame = threshold.process(frame)

    # turn depth data into a numpy array
    depth_image = np.asanyarray(frame.get_data())
    # colorize the depth data and turn it into a numpy array
    colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

    # generates a binary image showing objects within a given depth threshold to isolate the trees
    depth_mask = cv2.inRange(depth_image, distance_min * 1000, distance_max * 1000)
    depth_mask = cv2.bitwise_and(depth_mask, sky_binary)
    depth_mask[0:80, 0:depth_frame.width] = 0

    # runs closing algorithme on binary image
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # uses binary image as mask on color image, so it only shows the objects within the threshold
    masked = cv2.bitwise_and(color_image, color_image, mask=depth_mask)

    return colorized_depth, masked, depth_mask, thresh_frame


def cutTrunkAndGround(trunk, color_trunk_box):
    height, width = trunk.shape[:2]

    # Threshold for rectangles found in findTrunk
    inputImg_threshold = cv2.inRange(trunk, (254, 0, 0), (255, 0, 0))

    # Finds rectangles that mark the same trunk
    contours, hierarchy = cv2.findContours(inputImg_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_coord = []
    # Removes the trunks found
    for cnt in contours:
        x, y, cntWidth, cntHeight = cv2.boundingRect(cnt)
        cv2.rectangle(color_trunk_box, (x, y), (x + cntWidth, y + cntHeight), (120, 255, 0), 2)
        cv2.rectangle(trunk, (x, y), (x + cntWidth, y + cntHeight), (0, 0, 0), -1)
        box_coord.append((x + int(cntWidth / 2), y + int(cntHeight / 2)))

    # If a trunk is found, remove the ground under
    if len(box_coord) > 0:
        ground = int(sum([item[1] for item in box_coord]) / len(box_coord))
    else:
        ground = height

    # If multiple trunks is found, draw a line between them to separate tree crowns
    lineBetween = []
    box_coord.sort(reverse=True)
    while len(box_coord) > 1:
        first_coord = box_coord.pop(0)
        second_coord = box_coord[0]
        lineBetween.append(int(second_coord[0] + (first_coord[0] - second_coord[0]) / 2))
    for obj in lineBetween:
        cv2.rectangle(trunk, (obj - 5, 0), (obj + 5, height), (0, 0, 0), -1)

    cv2.rectangle(trunk, (0, ground), (width, height), (0, 0, 0), -1)

    trunk = cv2.cvtColor(trunk, cv2.COLOR_BGR2GRAY)

    return trunk, color_trunk_box


def findTrunk(binaryImage):
    # The binary image there is used as input is converted to BGR to make sure
    # it is possible to add bounding boxes in color later

    # Then a ROI is created to focus the template matching in the area where the trunks are located
    inputImg = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
    height, width = binaryImage.shape
    ROI = binaryImage[(height // 2)+20:height-80, 0:width]
    ROIh, ROIw = ROI.shape
    ROI = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)

    # Then to determined how many template the folder is run through and added the template to a list
    numberOfTemplates = 0
    templateList = list()
    for trunk in os.listdir("Trunks"):
        if path.isfile(path.join("Trunks", trunk)):
            numberOfTemplates += 1
            template = cv2.imread(f"Trunks\\Trunk{numberOfTemplates}.png")
            templateList.append(template)

    # Then the templates is used for template matching on the ROI one at the time using the SQDIFF NORMED method
    # Then for each match there is better than 0.28 it adds that location to a new list containing the best matching
    # The locations is added to the list together with the opposite corner so the bounding boxes can be created.
    boxes = list()
    for i in range(numberOfTemplates):
        H, W = templateList[i].shape[:2]
        outputTemplate = cv2.matchTemplate(ROI, templateList[i], cv2.TM_SQDIFF_NORMED)
        (y_points, x_points) = np.where(outputTemplate <= 0.28)

        outputTemplate = cv2.cvtColor(outputTemplate, cv2.COLOR_GRAY2BGR)

        for (x, y) in zip(x_points, y_points):
            box = (x, y, x + W, y + H)
            boxes.append(box)

    # Then non-max suppression is done on the found matches to eliminate the overlapping matches
    boxes = non_max_suppression(np.array(boxes), overlapThresh=0.5)

    # Then all the matches is printed onto the input image and then that image is returned.
    inputImg_C = inputImg.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(outputTemplate, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.rectangle(ROI, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.rectangle(inputImg_C, (x1 - 10, y1 + height - 80 - ROIh),
                      (x2 + 10, y2 + height - 80 - ROIh), (255, 0, 0), 3)
        # cv2.rectangle(inputImg_C, (0,((height // 2)+20)), (0+ROIw,((height // 2)+20+ROIh)), (255,50,125), 3)

    return inputImg_C


def findContours(closing_bgr, color_image, depth_frame, depth_intrinsics):
    # finds contours
    contours, hierarchy = cv2.findContours(closing_bgr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Creates copies so it does not mess with the originals
    closing_bgr_C = closing_bgr.copy()
    color_image_C = color_image.copy()
    closing_bgr_C = cv2.cvtColor(closing_bgr_C, cv2.COLOR_GRAY2BGR)
    # print(contours)

    depth_image = np.asanyarray(depth_frame.get_data())

    # runs through each objects
    for cnt in contours:
        # only detect the objects if the area is over a given pixel size
        if cv2.contourArea(cnt) > 2000:
            # finds bounding box for the objects
            x, y, width, height = cv2.boundingRect(cnt)

            # finds the avg. distance to the object
            box_dist = depth_image[y:y + height, x:x + width]
            box_dist = box_dist.reshape((box_dist.shape[0] * box_dist.shape[1], 1))
            box_dist = box_dist[np.nonzero(box_dist)]
            if len(box_dist):
                dist = (sum(box_dist) / len(box_dist)) / 1000
            else:
                dist = 0

            # finds real width/height, with pixel width/height, depth and focal length
            irlWidth = (dist * width) / depth_intrinsics.fx
            irlHeight = (dist * height) / depth_intrinsics.fy

            # position relative to sensor
            x_coord = (x - depth_intrinsics.ppx)
            y_coord = -(y - depth_intrinsics.ppy)
            irl_x = (dist * x_coord) / depth_intrinsics.fx
            irl_z = (dist * y_coord) / depth_intrinsics.fy

            # draws rectangle and writes information for the bounding box in binary image
            cv2.rectangle(closing_bgr_C, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.circle(closing_bgr_C, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(closing_bgr_C, f'Pixel Width: {width} & Pixel Height: {height}',
                        (x, y + height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(closing_bgr_C, f'Depth: {round(dist, 2)}m', (x, y + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(closing_bgr_C, f'Real Width: {round(irlWidth, 2)}m & Real Height: {round(irlHeight, 2)}m',
                        (x, y + height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(closing_bgr_C, f'Position x: {round(irl_x, 2)}m y: {round(dist, 2)}m z: {round(irl_z, 2)}m',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            # draws rectangle and writes information for the bounding box in color image
            cv2.rectangle(color_image_C, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.circle(color_image_C, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(color_image_C, f'Pixel Width: {width} & Pixel Height: {height}', (x, y + height + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(color_image_C, f'Depth: {round(dist, 2)}m', (x, y + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(color_image_C, f'Real Width: {round(irlWidth, 2)}m & Real Height: {round(irlHeight, 2)}m',
                        (x, y + height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(color_image_C, f'Position x: {round(irl_x, 2)}m y: {round(dist, 2)}m z: {round(irl_z, 2)}m',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    return closing_bgr_C, color_image_C


def imageShow(bag_file_run, video_done, color_box, depth_box, fps):
    # Displays the fps in the frame
    cv2.putText(depth_box, f'FPS: {round(fps, 2)}', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(color_box, f'FPS: {round(fps, 2)}', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    # color_box = cv2.resize(color_box, (int(848 * 1.5), int(480 * 1.5)), interpolation=cv2.INTER_AREA)

    # cv2.imshow("Binary", binary_image)
    # cv2.imshow("Color", color_image)
    # cv2.imshow("Trunk Box", trunk_box)
    # cv2.imshow("Depth Binary", depth_binary)
    # cv2.imshow("Depth Box", depth_box)
    cv2.imshow("Color Stream", color_box)
    cv2.imshow("Depth Stream", depth_box)

    # if pressed escape exit program
    key = cv2.waitKey(1)
    if video_done and not bag_file_run[1] and bag_file_run[2]:
        key = 27
    if key == 27:
        cv2.destroyAllWindows()
        if bag_file_run[2] and not bag_file_run[1]:
            main()
        exit()


def main():
    # Write the name of the file you want to run in the first argument in bagFileRun
    # The files can be found in the folder called trainingBagFiles
    bagFileRun = ("training1.bag", True, False)
    # If you want to use input in commandline, then set the second argument in bagFileRun to False
    # if you want to loop the script when using input, to run through different bag files. Set last argument to True

    # This function initializes the pipline
    pipeline, frameNumberStart = initialize(bagFileRun)

    fps = 0
    while True:
        start_time = time.time()
        # This function pulls the frames from the pipeline
        depth_frame, colorized_depth, color_image, videoDone, depth_intrinsics = getFrames(pipeline, frameNumberStart)

        # Process color data and isolates objects within a given color threshold
        minThresh = np.array([230, 230, 230])  # ([minB, minG, minR])
        maxThresh = np.array([255, 255, 255])  # ([maxB, maxG, maxR])
        removeSky, sky_binary = colorThresholding(color_image, minThresh, maxThresh, kernel=np.ones((3, 3), np.uint8))

        # Process depth data and isolates objects within a given depth threshold
        modified_colorized_depth, color_removed_background, depth_masked, depth_image = \
            removeBackground(depth_frame, color_image, sky_binary, distance_max=4,
                             distance_min=0.2)  # distance is in meters

        # Detects trunks in the binary image
        trunk_box = findTrunk(depth_masked)

        # Uses trunk_box to cut trunk and ground
        treeCrown_box, color_trunk_box = cutTrunkAndGround(trunk_box, color_image)

        # Detects tree crowns and calculates real world width, height and position relative to the camera
        depth_masked_trunk_box, color_image_box = findContours(treeCrown_box, color_image, depth_image,
                                                               depth_intrinsics)

        # Render images in opencv window
        imageShow(bagFileRun, videoDone, color_image_box, depth_masked_trunk_box, fps)

        # Calculate fps
        end_time = time.time() - start_time
        fps = 1 / end_time


if __name__ == "__main__":
    main()
