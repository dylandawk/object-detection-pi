# object-detection-pi
Object detection program for raspberry pi adopted from Google media pipe example. https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/object_detection/raspberry_pi

## Set up your hardware

Before you begin, you need to
[set up your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
with Raspberry 64-bit Pi OS (preferably updated to Buster).

You also need to [connect and configure the Pi Camera](
https://www.raspberrypi.org/documentation/configuration/camera.md) if you use
the Pi Camera. This code also works with USB camera connect to the Raspberry Pi.

And to see the results from the camera, you need a monitor connected
to the Raspberry Pi. It's okay if you're using SSH to access the Pi shell
(you don't need to use a keyboard connected to the Pi)—you only need a monitor
attached to the Pi to see the camera stream.

## Set up virtual environment

```
cd object-detection-pi
python -m venv --system-site-packages env
source env/bin/activate
```

## Install MediaPipe

Run this script to install the required dependencies and download the TFLite models:

```
sh setup.sh
```

## Run the example

```
python3 simple_detect.py 
```

You should see the camera feed appear on the monitor attached to your Raspberry
Pi. Put some objects in front of the camera, like a coffee mug or keyboard, and
you'll see boxes drawn around those that the model recognizes, including the
label and score for each. It also prints the number of frames per second (FPS)
at the top-left corner of the screen. As the pipeline contains some processes
other than model inference, including visualizing the detection results, you can
expect a higher FPS if your inference pipeline runs in headless mode without
visualization.

*   You can optionally specify the `model` parameter to set the TensorFlow Lite
    model to be used:
    *   The default value is `efficientdet_lite0.tflite`
    *   TensorFlow Lite object detection models **with metadata**  
        * Models from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models)
        * Models trained with [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/customization/object_detector) are supported.
*   You can optionally specify the `maxResults` parameter to limit the list of
    detection results:
    *   Supported value: A positive integer.
    *   Default value: `5`
*   You can optionally specify the `scoreThreshold` parameter to adjust the
    score threshold of detection results:
    *   Supported value: A floating-point number.
    *   Default value: `0.25`.
*   Example usage:
    ```
    python3 detect.py \
      --model efficientdet_lite0.tflite \
      --maxResults 5 \
      --scoreThreshold 0.3
    ```
