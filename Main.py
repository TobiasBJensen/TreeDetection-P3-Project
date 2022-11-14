import pyrealsense2 as rs
import numpy as np
import cv2
from sys import platform
from os import path


def pathToFile():
    #bagFile = input("Input Bagfile: ")
    # If you want to run the same file a lot just uncomment the line below and write the name of the file
    bagFile = "20221110_142511"

    if platform == "win32":
        pathToBag = f"trainingBagFiles\\{bagFile}.bag"
        if not path.isfile(pathToBag):
            pathToBag = f"D:\\Rob3_Gruppe_6_Realsense_data\\BagfileTest\\{bagFile}.bag"

    if platform == "darwin":
        pathToBag = f"trainingBagFiles/{bagFile}.bag"
        if not path.isfile(pathToBag):
            pathToBag = f"D:/Rob3_Gruppe_6_Realsense_data/BagfileTest/{bagFile}.bag"

    if not path.isfile(pathToBag):
        print("Can't find a file with that name")
        pathToFile()

    else:
        return pathToBag


distance_max = 6  # meter
distance_min = 0.2  # meter

try:
    # Path towards a bag file
    pathToRosBag = pathToFile()

    # Align RGB to depth
    align = rs.align(rs.stream.depth)
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
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object for the depth stream
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        # Get depth frame and color frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Colorize depth frame to jet colormap

        # Fill holes in depth dataset
        hole_filling = rs.hole_filling_filter(2)
        filled_depth = hole_filling.process(depth_frame)
        #filled_depth = depth_frame

        depth_color_frame = colorizer.colorize(filled_depth)
        # Convert depth_frame to numpy array to render image in opencv
        depth_image = np.asanyarray(filled_depth.get_data())
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        depth_gray = cv2.cvtColor(depth_color_image, cv2.COLOR_RGB2GRAY)

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        depth_image = cv2.GaussianBlur(depth_image, (51, 51), 0)
        mask = cv2.inRange(depth_image, distance_min * 1000, distance_max * 1000)
        masked = cv2.bitwise_and(color_image, color_image, mask=mask)

        cv2.imshow("Color Stream", masked)
        cv2.imshow("Depth Stream", depth_color_image)

        # if pressed escape exit program
        key = cv2.waitKey(1)
        if key == 27:  # esc
            cv2.destroyAllWindows()
            break

except RuntimeError:
    print("Can't read the given file, are you sure it is the right type?")
    pathToFile()

finally:
    pass
