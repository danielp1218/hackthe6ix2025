import cv2
import mediapipe as mp
import numpy as np

def detect_poses():
    # mediapipe stuffs
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: cant open cameras")
        return
        
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error: cant read frames")
            break
        
        frames = [cv2.resize(frame1, (320, 240)), cv2.resize(frame2, (320, 240))]
        for i, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                landmarks = results.pose_landmarks.landmark

                h, w, c = frame.shape
                
                x_coords = [landmark.x * w for landmark in landmarks]
                y_coords = [landmark.y * h for landmark in landmarks]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add some padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {i+1}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow(f'Pose Detection Camera {i+1}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    detect_poses()
