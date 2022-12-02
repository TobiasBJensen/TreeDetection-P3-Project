# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import os
from os import path
import pickle


# write list to binary file
def write_list(a_list):
    # store list in binary file so 'wb' mode
    with open('listfile', 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


def main():
    try:
        runOnce = True
        bagfile = input("Input bagfile: ")
        # Path towards a bag file
        pathToRosBag = f"trainingBagFiles\\{bagfile}.bag"
        if not path.isfile(pathToRosBag):
            pathToRosBag = f"D:\\Rob3_Gruppe_6_Realsense_data\\BagfileTest\\{bagfile}.bag"
            if not path.isfile(pathToRosBag):
                pathToRosBag = f"D:\\Rob3_Gruppe_6_Realsense_data\\BagfileTest\\Test\\{bagfile}.bag"
        if not pathToRosBag:
            print("The given file can not be found")
            main()
        if bagfile == "exit":
            exit()

        # Create opencv window to render image in
        # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)

        align = rs.align(rs.stream.depth)
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

        # Create colorizer object for the depth stream
        colorizer = rs.colorizer()

        for i in range(5):
            if i == 4:
                pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        startNumber = frames.get_frame_number() + 1
        frameSet = []

        # Streaming loop
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
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
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Render image in opencv window
            #cv2.imshow("Depth Stream", depth_color_image)
            cv2.imshow("Color Stream", color_image)

            # if pressed escape exit program
            key = cv2.waitKey(1)
            currentNumber = frames.get_frame_number()
            #print(startNumber)
            #print(currentNumber)
            frame = color_image, currentNumber
            frameSet.append(frame)

            if currentNumber <= startNumber and runOnce:
                #print("hit")
                break

            if key == 27:
                break

        write_list(frameSet)

        num = 1
        pause = True
        for frame in frameSet:
            cv2.imshow("Color Stream", frame[0])
            print(frame[1])
            # esc exit, 's' start and 'p'
            while pause:
                key = cv2.waitKey(1)
                if key == ord('s'):
                    pause = False

                if key == ord('c'):
                    #cv2.imwrite(os.path.join('test', f'tree{num}.jpg'), frame[0])
                    pause = False
                    num = num + 1

            key = cv2.waitKey(50)
            if key == ord('p'):
                pause = True

            if key == 27:
                cv2.destroyAllWindows()
                exit()

    except RuntimeError:
        print("Try again")
        main()

    finally:
        pass


if __name__ == "__main__":
    main()
