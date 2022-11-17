import pyrealsense2 as rs
import numpy as np
import cv2 as cv

try:
    # Path towards a bag file
    pathToRosBag = "20221104_125803.bag"



    # Create pipeline
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
    cv.namedWindow("Depth Stream", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Color Stream", cv.WINDOW_AUTOSIZE)

    # Create colorizer object for the depth stream
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()

        # Get depth frame and color frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert colorized depth and RGB color frames to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Remeber to convert from RGB color format to OpenCV BGR color format
        # Without converting from one format to another, colors are not visualised correctly
        # (check out the sponge on the shelf in the color stream with and without converting)
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)

        # Render image in opencv window
        cv.imshow("Depth Stream", depth_color_image)
        cv.imshow("Color Stream", color_image)

        key = cv.waitKey(0)  # Hvis sættes til 1 bliver det video eller kommer der et frame af gangen
        # if pressed escape exit program

        # if pressed escape exit program
        if key == 27:
            cv.destroyAllWindows()
            break

finally:
    pass


