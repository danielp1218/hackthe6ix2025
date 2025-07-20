import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import socket
import json
import struct
import threading
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

GAME_SOCKET_PORT = 5000
GAME_SOCKET_IP = '10.33.11.89'

# UDP receiver configuration
UDP_RECEIVER_PORT = 9999
UDP_RECEIVER_IP = '10.33.53.104'

# Frame header structure (must match C code)
FRAME_HEADER_FORMAT = '<IIIIIIII'  # little-endian: magic, camera_id, sequence, frametype, width, height, stride, data_size
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FORMAT)
FRAME_MAGIC = 0x46524D45  # "FRME"

# Game UDP socket
game_socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
game_socket_client.bind((GAME_SOCKET_IP, GAME_SOCKET_PORT))

# game_socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Note: We don't bind here since we're only sending
print(f"Game UDP client ready to send to {GAME_SOCKET_IP}:{GAME_SOCKET_PORT}")

frame_shape = [720, 1280]

# Add here if you need more keypoints
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

class UDPFrameReceiver:
    def __init__(self, port=UDP_RECEIVER_PORT, ip=UDP_RECEIVER_IP):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((ip, port))
        self.socket.settimeout(1.0)  # 1 second timeout
        print(f"UDP receiver listening on {ip}:{port}")
        
        # Frame buffers for each camera - store multiple recent frames
        self.frame_buffers = {0: deque(maxlen=5), 1: deque(maxlen=5)}  # Keep last 5 frames
        self.frame_locks = {0: threading.Lock(), 1: threading.Lock()}
        
        # Frame assembly buffers
        self.frame_assembly = {}  # key: (camera_id, sequence), value: {'header': header, 'data': bytearray, 'received': int}
        
        self.running = True
        self.receiver_thread = threading.Thread(target=self._receive_frames)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
    
    def _receive_frames(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)  # Max UDP packet size
                
                if len(data) == FRAME_HEADER_SIZE:
                    # This is a header packet
                    self._handle_header(data)
                else:
                    # This is frame data
                    self._handle_frame_data(data, addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP receiver error: {e}")
                continue
    
    def _handle_header(self, data):
        try:
            header = struct.unpack(FRAME_HEADER_FORMAT, data)
            magic, camera_id, sequence, frametype, width, height, stride, data_size = header
            
            if magic != FRAME_MAGIC:
                print(f"Invalid frame magic: {magic:08x}")
                return
            
            # Initialize frame assembly buffer
            frame_key = (camera_id, sequence)
            self.frame_assembly[frame_key] = {
                'header': header,
                'data': bytearray(),
                'received': 0,
                'expected_size': data_size
            }
            
        except Exception as e:
            print(f"Header parsing error: {e}")
    
    def _handle_frame_data(self, data, addr):
        # Find the most recent frame assembly for any camera that's expecting data
        for frame_key, assembly in list(self.frame_assembly.items()):
            if assembly['received'] < assembly['expected_size']:
                # Add data to this frame
                assembly['data'].extend(data)
                assembly['received'] += len(data)
                
                # Check if frame is complete
                if assembly['received'] >= assembly['expected_size']:
                    self._process_complete_frame(frame_key, assembly)
                    del self.frame_assembly[frame_key]
                break
    
    def _process_complete_frame(self, frame_key, assembly):
        camera_id, sequence = frame_key
        header = assembly['header']
        magic, camera_id, sequence, frametype, width, height, stride, data_size = header
        
        print(f"Processing frame - Camera: {camera_id}, Type: {frametype}, Size: {width}x{height}, Data: {len(assembly['data'])} bytes")
        
        try:
            # Decode frame based on frametype
            if frametype == 99:  # Grayscale
                frame_data = self._decode_grayscale(assembly['data'], width, height)
            elif frametype == 100:  # RLE compressed grayscale
                frame_data = self._decode_rle_grayscale(assembly['data'], width, height)
            else:
                print(f"Unsupported frametype: {frametype}")
                return
            
            # Convert to OpenCV format (BGR)
            if len(frame_data.shape) == 2:  # Grayscale
                frame_bgr = cv.cvtColor(frame_data, cv.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame_data
            
            # Resize to expected frame shape (720x720)
            frame_resized = cv.resize(frame_bgr, (frame_shape[0], frame_shape[0]))
            
            # Store frame with timestamp in buffer
            with self.frame_locks[camera_id]:
                frame_with_timestamp = {
                    'frame': frame_resized,
                    'timestamp': time.time(),
                    'sequence': sequence
                }
                self.frame_buffers[camera_id].append(frame_with_timestamp)
                
        except Exception as e:
            print(f"Frame processing error for camera {camera_id}: {e}")
            print(f"Debug info - Expected size: {width*height}, Actual data: {len(assembly['data'])}, Frame type: {frametype}")
    
    def _decode_grayscale(self, data, width, height):
        # Simple grayscale frame
        expected_size = width * height
        if len(data) < expected_size:
            print(f"Warning: Not enough data. Expected {expected_size}, got {len(data)}")
            # Pad with zeros if data is too short
            padded_data = bytearray(data)
            padded_data.extend([0] * (expected_size - len(data)))
            frame_array = np.frombuffer(padded_data[:expected_size], dtype=np.uint8)
        else:
            frame_array = np.frombuffer(data[:expected_size], dtype=np.uint8)
        return frame_array.reshape((height, width))
    
    def _decode_rle_grayscale(self, data, width, height):
        # RLE decompression
        decoded = bytearray()
        i = 0
        expected_size = width * height
        
        while i < len(data) and len(decoded) < expected_size:
            if data[i] == 0 and i + 2 < len(data):  # RLE marker
                count = data[i + 1]
                value = data[i + 2]
                decoded.extend([value] * count)
                i += 3
            else:
                decoded.append(data[i])
                i += 1
        
        # Ensure we have the right amount of data
        if len(decoded) < expected_size:
            print(f"RLE decode warning: Expected {expected_size}, got {len(decoded)}")
            decoded.extend([0] * (expected_size - len(decoded)))
        
        # Convert to numpy array
        frame_array = np.frombuffer(decoded[:expected_size], dtype=np.uint8)
        return frame_array.reshape((height, width))
    
    def get_frame(self, camera_id):
        """Get the latest frame for a specific camera"""
        with self.frame_locks[camera_id]:
            if len(self.frame_buffers[camera_id]) > 0:
                latest_frame = self.frame_buffers[camera_id][-1]  # Get most recent
                return latest_frame['frame'].copy()
            return None
    
    def get_frames(self, max_time_diff=0.2):
        """Get the best synchronized frames from both cameras"""
        with self.frame_locks[0], self.frame_locks[1]:
            # Check if we have frames from both cameras
            if len(self.frame_buffers[0]) == 0 or len(self.frame_buffers[1]) == 0:
                return None, None
            
            # Find the best matching frame pair based on timestamps
            best_pair = None
            best_time_diff = float('inf')
            
            # Look through recent frames to find best temporal match
            for frame0_data in reversed(list(self.frame_buffers[0])):  # Most recent first
                for frame1_data in reversed(list(self.frame_buffers[1])):
                    time_diff = abs(frame0_data['timestamp'] - frame1_data['timestamp'])
                    if time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_pair = (frame0_data, frame1_data)
                        
                        # If we found a very good match, stop searching
                        if time_diff < 0.05:  # 50ms is very good sync
                            break
                if best_time_diff < 0.05:
                    break
            
            if best_pair and best_time_diff < max_time_diff:
                frame0_data, frame1_data = best_pair
                print(f"Using frames with time diff: {best_time_diff:.3f}s")
                return frame0_data['frame'].copy(), frame1_data['frame'].copy()
            elif best_pair:
                print(f"Frame sync warning: time difference {best_time_diff:.3f}s > {max_time_diff}s")
                frame0_data, frame1_data = best_pair
                return frame0_data['frame'].copy(), frame1_data['frame'].copy()
            
            return None, None
    
    def stop(self):
        self.running = False
        self.receiver_thread.join()
        self.socket.close()


def run_mp_udp(P0, P1):
    # Initialize UDP frame receiver
    receiver = UDPFrameReceiver()
    
    # Create body keypoints detector objects
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    print("Waiting for UDP frames from cameras...")
    
    try:
        while True:
            sendable = True

            # Get frames from UDP receiver
            frame0, frame1 = receiver.get_frames()
            
            if frame0 is None or frame1 is None:
                time.sleep(0.01)  # Wait for frames
                continue

            # Process frames (same as original code)
            # Convert BGR to RGB for MediaPipe
            frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
            frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable
            frame0_rgb.flags.writeable = False
            frame1_rgb.flags.writeable = False
            results0 = pose0.process(frame0_rgb)
            results1 = pose1.process(frame1_rgb)

            # Reverse changes
            frame0_rgb.flags.writeable = True
            frame1_rgb.flags.writeable = True

            # Check for keypoints detection
            frame0_keypoints = []
            if results0.pose_landmarks:
                for i, landmark in enumerate(results0.pose_landmarks.landmark):
                    if i not in pose_keypoints: continue
                    pxl_x = landmark.x * frame0.shape[1]
                    pxl_y = landmark.y * frame0.shape[0]
                    pxl_x = int(round(pxl_x))
                    pxl_y = int(round(pxl_y))
                    cv.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                    kpts = [pxl_x, pxl_y]
                    frame0_keypoints.append(kpts)
            else:
                frame0_keypoints = [[-1, -1]] * len(pose_keypoints)

            frame1_keypoints = []
            if results1.pose_landmarks:
                for i, landmark in enumerate(results1.pose_landmarks.landmark):
                    if i not in pose_keypoints: continue
                    pxl_x = landmark.x * frame1.shape[1]
                    pxl_y = landmark.y * frame1.shape[0]
                    pxl_x = int(round(pxl_x))
                    pxl_y = int(round(pxl_y))
                    cv.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                    kpts = [pxl_x, pxl_y]
                    frame1_keypoints.append(kpts)
            else:
                frame1_keypoints = [[-1, -1]] * len(pose_keypoints)
                sendable = False

            # Calculate 3d position
            frame_p3ds = []
            for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
                if uv1[0] == -1 or uv2[0] == -1:
                    _p3d = [-1, -1, -1]
                else:
                    _p3d = DLT(P0, P1, uv1, uv2)
                frame_p3ds.append(_p3d)

            print(f"Sent {frame_p3ds} to server.")

            frame_p3ds = np.array(frame_p3ds).reshape((12, 3))

            if sendable:
                game_socket_client.sendto(json.dumps(frame_p3ds.tolist()).encode(), (GAME_SOCKET_IP, GAME_SOCKET_PORT))

            print(frame_p3ds)

            # Real time 3D visualization
            ax.clear()
            torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
            armr = [[1, 3], [3, 5]]
            arml = [[0, 2], [2, 4]]
            legr = [[6, 8], [8, 10]]
            legl = [[7, 9], [9, 11]]
            body = [torso, arml, armr, legr, legl]
            colors = ['red', 'blue', 'green', 'black', 'orange']

            for bodypart, part_color in zip(body, colors):
                for _c in bodypart:
                    ax.plot(xs=[frame_p3ds[_c[0], 0], frame_p3ds[_c[1], 0]],
                            ys=[frame_p3ds[_c[0], 1], frame_p3ds[_c[1], 1]],
                            zs=[frame_p3ds[_c[0], 2], frame_p3ds[_c[1], 2]],
                            linewidth=4, c=part_color)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.set_xlim3d(-500, 1000)
            ax.set_xlabel('x')
            ax.set_ylim3d(-500, 1000)
            ax.set_ylabel('y')
            ax.set_zlim3d(-500, 1000)
            ax.set_zlabel('z')
            plt.pause(0.1)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv.imshow('cam1', frame1)
            cv.imshow('cam0', frame0)

            k = cv.waitKey(1)
            if k & 0xFF == 27: break  # ESC key

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        receiver.stop()
        cv.destroyAllWindows()


if __name__ == '__main__':
    # Get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    run_mp_udp(P0, P1)
