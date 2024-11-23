"""Main scripts to run object detection."""

import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Global variables to determine person
PERSON = "person"
NONE_OBJ = "NONE"
# Time in secs before resetting
RESET_LENGTH = 5.0


class SimpleDetector(object):

    def __init__(self, 
                 model, 
                 maxResults, 
                 scoreThreshold, 
                 cameraId, 
                 frameWidth, 
                 frameHeight,
                 visualize
        ):
        self.model = model
        self.maxResults = maxResults
        self.scoreThreshold = scoreThreshold
        self.cameraId = cameraId 
        self.frameWidth = frameWidth 
        self.frameHeight = frameHeight
        self.start_time = time.time()
        self.timer_reset = False
        self.closed = True
        self.visualize = visualize
    
    def run(self) -> None:
        """Continuously run inference on images acquired from the camera.

        Args:
            model: Name of the TFLite object detection model.
            max_results: Max number of detection results.
            score_threshold: The score threshold of detection results.
            camera_id: The camera id to be passed to OpenCV.
            width: The width of the frame captured from the camera.
            height: The height of the frame captured from the camera.
        """
        

        # Start capturing video input from the camera
        cap = cv2.VideoCapture(int(self.cameraId))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)

        # Visualization parameters
        row_size = 50  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 0)  # black
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        detection_frame = None
        detection_result_list = []

        # timer start time
        begin_time = time.time()

        def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
            global FPS, COUNTER, START_TIME

            # Calculate the FPS
            if COUNTER % fps_avg_frame_count == 0:
                FPS = fps_avg_frame_count / (time.time() - START_TIME)
                START_TIME = time.time()

            detection_result_list.append(result)
            COUNTER += 1
        
        # Initialize the object detection model
        base_options = python.BaseOptions(
            model_asset_path=self.model  
        )
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                max_results=self.maxResults, 
                                                score_threshold=self.scoreThreshold,
                                                category_allowlist=[PERSON],
                                                result_callback=save_result)
        detector = vision.ObjectDetector.create_from_options(options)

        current_object_name = None
        # Continuously capture images from the camera and run inference
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Run object detection using the model.
            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            current_frame = image
            if detection_result_list:
                if self.visualize:
                    # Prepare object detection image visual
                    current_frame = visualize(current_frame, detection_result_list[0])
                    detection_frame = current_frame
                # Iterate through the detection results       
                for detected_object in detection_result_list[0].detections:
                    detected_object_name = detected_object.categories[0].category_name
                    # If the object has changed run callback
                    if current_object_name != detected_object_name:
                        current_object_name = detected_object_name
                        self.on_object_changed(current_object_name)
                detection_result_list.clear()
            else:
                # If no object detect when it was detected in previous result, run callback
                if current_object_name != NONE_OBJ:
                    current_object_name = NONE_OBJ
                    self.on_object_changed(current_object_name)
                # If the timer has been reset check the amount of time that has passed since reset
                if self.timer_reset == True:
                    # If the reset time is above the RESET_LENGTH stop the timer and trigger close
                    if time.time() - self.start_time > RESET_LENGTH:
                        print("Closing Flower!")
                        self.timer_reset = False
                        self.closed = True
                    
            # Display the image
            if self.visualize and detection_frame is not None:
                cv2.imshow('object_detection', detection_frame)

            # Stop the program if the ESC key is pressed.
            if cv2.waitKey(1) == 27:
                break

        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    
    def on_object_changed(self, obj_name: str):
        # print(f"Object changed: {obj_name}")
        if obj_name == NONE_OBJ:
            self.start_time = time.time()
            self.timer_reset = True
        elif obj_name == PERSON:
            if self.closed:
                self.closed = False
                print("Opening Flower!")

    
def check_cams():
    for i in range(0, 3):
        cap = cv2.VideoCapture(i)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            print(f"Index: {i} is good")
            break
        cap.release()
            

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='ssdmobilenet.tflite')
    parser.add_argument(
        '--maxResults',
        help='Max number of detection results.',
        required=False,
        default=1)
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of detection results.',
        required=False,
        type=float,
        default=0.7)
    # Finding the camera ID can be very reliant on platform-dependent methods. 
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0. 
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId',
        help='Id of camera.', 
        required=False, 
        type=int, 
        default=0,
    )
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=320
    )
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=240
    )
    parser.add_argument(
        '--visualize',
        help='Display detection results in UI',
        action='store_true',
    )
    args = parser.parse_args()

    simple_detector = SimpleDetector(
        model = args.model,
        maxResults=int(args.maxResults),
        scoreThreshold=args.scoreThreshold,
        cameraId=int(args.cameraId),
        frameWidth=args.frameWidth,
        frameHeight=args.frameHeight,
        visualize=args.visualize,
    )
    simple_detector.run()


if __name__ == '__main__':
    main()
    # check_cams()
