import pyrealsense2 as rs
import numpy as np
import cv2
from sys import platform
from os import path


def pathToFile():
    bagFile = input("Input Bagfile: ")
    # If you want to run the same file a lot just uncomment the line below and write the name of the file
    #bagFile = "20221110_142511"

    if platform == "win32":
        pathToBag = f"trainingBagFiles\\{bagFile}"
        if not path.isfile(pathToBag):
            pathToBag = f"D:\\Rob3_Gruppe_6_Realsense_data\\BagfileTest\\{bagFile}"

    if platform == "darwin":
        pathToBag = f"trainingBagFiles/{bagFile}"
        if not path.isfile(pathToBag):
            pathToBag = f"D:/Rob3_Gruppe_6_Realsense_data/BagfileTest/{bagFile}"

    if not path.isfile(pathToBag):
        print("Can't find a file with that name")
        pathToFile()

    else:
        return pathToBag


def initialize():
    try:
        # Path towards a bag file
        pathToRosBag = pathToFile()

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


    except RuntimeError:
        print("Can't read the given file, are you sure it is the right type?")
        pathToFile()
    finally:
        return pipeline


def getFrames(pipeline):
    # Create colorizer object for the depth stream
    colorizer = rs.colorizer()
    # Align RGB to depth
    alignD = rs.align(rs.stream.depth)
    alignC = rs.align(rs.stream.color)

    for x in range(5):
        pipeline.wait_for_frames()


    # Get frames
    frameset = pipeline.wait_for_frames()
    frameset = alignD.process(frameset)
    depth_frame = frameset.get_depth_frame()
    color_frame = frameset.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    depth_image = np.asanyarray(depth_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(depth_image).get_data())

    return depth_frame, colorized_depth, color_image

def removeBackground(depth_frame, color_image):
    distance_max = 4  # meter
    distance_min = 0  # meter

    colorizer = rs.colorizer()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter(2)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.3)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)

    frame = depth_to_disparity.process(depth_frame)
    frame = spatial.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

    depth_image = np.asanyarray(frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

    mask = cv2.inRange(depth_image, distance_min * 1000, distance_max * 1000)
    masked = cv2.bitwise_or(color_image, color_image, mask=mask)

    return colorized_depth, masked
def main():
    pipeline = initialize()

    while True:
        depth_frame, colorized_depth, color_image = getFrames(pipeline)
        modified_colorized_depth, color_removed_background = removeBackground(depth_frame, color_image)
        # Render image in opencv window
        cv2.imshow("Depth Stream", modified_colorized_depth)
        cv2.imshow("Color Stream", color_removed_background)
        # if pressed escape exit program
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
