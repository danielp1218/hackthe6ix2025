import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from socket import *

parser = argparse.ArgumentParser(description="Run MediaPipe Pose Detection and send keypoints over UDP.")
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--camera', type=int, default=0, help='Input video stream (default: 0 for webcam)')
args = parser.parse_args()

camera_id = args.camera

# Define server connection parameters
SERVER_IP = args.ip
SERVER_PORT = args.port

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [720, 1280]



sock = socket(AF_INET, SOCK_DGRAM)
sock.connect((SERVER_IP, SERVER_PORT))

#add here if you need more keypoints
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

def run_mp(input_stream: int = 0):
    #input video stream
    cap = cv.VideoCapture(input_stream)

    cap.set(3, frame_shape[1])
    cap.set(4, frame_shape[0])

    #create body keypoints detector objects.
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:

        #read frames from stream
        ret, frame = cap.read()

        if not ret: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame.shape[1] != 720:
            frame = frame[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = pose.process(frame)
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        #check for keypoints detection
        frame_keypoints = []
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i not in pose_keypoints: continue #only save keypoints that are indicated in pose_keypoints
                pxl_x = landmark.x * frame.shape[1]
                pxl_y = landmark.y * frame.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame,(pxl_x, pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
                kpts = [pxl_x, pxl_y]
                frame_keypoints.append(kpts)
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame_keypoints = [[-1, -1]]*len(pose_keypoints)

        sock.sendall(json.dumps({"camera": camera_id, "keypoints": frame_keypoints}).encode())

while True:
    try:
        run_mp(camera_id)
    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break
