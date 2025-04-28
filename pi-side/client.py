import cv2
import requests
import os
import time
from datetime import datetime
import numpy as np

SERVER_URL = "http://3.15.203.82/upload"
IMAGE_PATH = "pose_prediction.png"
LAST_IMAGE_PATH = "screenshot_last.png"


video_capture = cv2.VideoCapture(0)

shape = (640, 480)
frame_resized = video_capture.read()[1]
last_frame_resized = frame_resized

def capture_screenshot(first_frame):
    global frame_resized
    global last_frame_resized
    
    if first_frame:
        save_path = LAST_IMAGE_PATH
    else:
        save_path = IMAGE_PATH
        
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return False

    # print(f"Captured Image: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    result, frame = video_capture.read() 
    if result:
        # cv2.imwrite(IMAGE_PATH, frame) 
        last_frame_resized = frame_resized
        
        frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480 resolution
        cv2.imwrite(save_path, frame_resized)
        cv2.imwrite(LAST_IMAGE_PATH, last_frame_resized)
                
        # print(f"Screenshot saved as {save_path}")
    else:
        print("Error: Could not capture image.")
    
   # video_capture.release()
    # print(f"Saved Image: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    return result

def upload_image(image_path):
    if not os.path.exists(image_path):
        # print(f"Error: File {image_path} not found!")
        return
    
    with open(image_path, "rb") as file:
        files = {"file": (os.path.basename(image_path), file, "image/png")}
        try:
            response = requests.post(SERVER_URL, files=files)
            response.raise_for_status() 
            # print("Server Response:", response.json())
            # print(f"ImageSent: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            # print("\n")
        except requests.exceptions.RequestException as e:
            print(f"Upload failed: {e}")

def mse():
    img_current = cv2.imread(IMAGE_PATH)
    img_last = cv2.imread(LAST_IMAGE_PATH)
    
    img_current_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    img_last_gray = cv2.cvtColor(img_last, cv2.COLOR_BGR2GRAY)
        
    err = np.sum((img_current_gray.astype("float") - img_last_gray.astype("float")) ** 2)
    err /= float(img_current_gray.shape[0] * img_current_gray.shape[1])
    
    return err

if __name__ == "__main__":
    capture_screenshot(True)
    while True:  
        # print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if capture_screenshot(False):
            if mse() > 150:
                print("Mean Square Error: ", mse())
                upload_image(IMAGE_PATH)
        # print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        time.sleep(.5)  

