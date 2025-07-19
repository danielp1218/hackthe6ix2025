import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import write_keypoints_to_disk
import socket
import json

frame_shape = [720, 1280]

#add here if you need more keypoints
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# this is stolen from show_3d_post.py LOL
def visualize_3d_live(frame_p3ds):
    """Visualizes 3D keypoints live."""
    print(frame_p3ds)
    global fig, ax

    torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    ax.clear()  # Clear the previous frame

    for bodypart, part_color in zip(body, colors):
        for _c in bodypart:
            ax.plot(xs=[frame_p3ds[_c[0], 0], frame_p3ds[_c[1], 0]],
                    ys=[frame_p3ds[_c[0], 1], frame_p3ds[_c[1], 1]],
                    zs=[frame_p3ds[_c[0], 2], frame_p3ds[_c[1], 2]],
                    linewidth=4, c=part_color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim3d(-300, 300)
    ax.set_xlabel('x')
    ax.set_ylim3d(0, 600)
    ax.set_ylabel('y')
    ax.set_zlim3d(200, 900)
    ax.set_zlabel('z')

    plt.pause(0.1)

def run_udp_server():
    """Main function to receive UDP packets and visualize 3D pose in real time."""
    # Create UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('127.0.0.1', 8080))
    udp_socket.settimeout(0.1)  # Non-blocking with timeout
    
    print("UDP server listening on 127.0.0.1:8080")
    print("Press ESC to exit...")
    
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    
    try:
        while True:
            try:
                # Receive UDP packet
                data, addr = udp_socket.recvfrom(8080)
                received_data = json.loads(data.decode())
                
                # Parse the received data (assuming it contains 3D keypoints)
                if isinstance(received_data, list) and len(received_data) > 0:
                    frame_p3ds = np.array(received_data).reshape((12, 3))
                    
                    # Store for later use
                    kpts_3d.append(frame_p3ds)
                    
                    # Visualize the received 3D pose
                    visualize_3d_live(frame_p3ds)
                    
            except socket.timeout:
                # Continue loop if no data received within timeout
                continue
            except json.JSONDecodeError:
                print("Error decoding JSON data")
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error receiving data: {e}")
                continue
                
            # Check for ESC key press (this requires a window to be active)
            if cv.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        udp_socket.close()
        cv.destroyAllWindows()
    
    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':
    # Start UDP server to receive packets from pose_detection.py
    kpts_cam0, kpts_cam1, kpts_3d = run_udp_server()

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)