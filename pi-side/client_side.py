import cv2
import requests
import numpy as np
import threading
import time
from datetime import datetime
from io import BytesIO
from PIL import Image

SERVER_UPLOAD_URL = "http://3.15.203.82/upload"
SERVER_FEEDBACK_URL = "http://3.15.203.82/json"
SERVER_POSE_IMAGE_URL = "http://3.15.203.82/files/pose_prediction.jpg"

IMAGE_PATH = "screenshot.jpg"
LAST_IMAGE_PATH = "screenshot_last.jpg"


video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

feedback_text = "Fetching feedback..."
pose_prediction_image = np.zeros((480, 640, 3), dtype=np.uint8) 
feedback_lock = threading.Lock()
pose_lock = threading.Lock()

"""
first_frame is a boolean: Only used if this is the first image of the session

This method will read the video capture from the webcam in order to extract 
the frame that we are going to send. Once it has this, it will resize the 
image to make it smaller (640 x 480) and saves it locally
"""
def capture_screenshot(first_frame):
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not capture image.")
        return False

    resized_frame = cv2.resize(frame, (640, 480))
    if first_frame:
        cv2.imwrite(LAST_IMAGE_PATH, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    else:
        last_frame = cv2.imread(IMAGE_PATH)
        if last_frame is not None:
            cv2.imwrite(LAST_IMAGE_PATH, last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        cv2.imwrite(IMAGE_PATH, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    return True
    
"""
Function that uploads the saved image to the AWS server

It will first get the filepath of the image locally,
then it will use the 'requests' library to upload the image
"""
def upload_image():
    if not IMAGE_PATH:
        return

    try:
        with open(IMAGE_PATH, "rb") as file:
            files = {"file": (IMAGE_PATH, file, "image/jpeg")}
            response = requests.post(SERVER_UPLOAD_URL, files=files, timeout=3)
            response.raise_for_status()
    except Exception as e:
        print(f"Upload failed: {e}")

"""
Function that retrieves the feedback from the AWS server
that was uploaded by the processing server

Uses the requests libray to fetch the data and parse the JSON
"""

def fetch_feedback():
    global feedback_text
    while True:
        try:
            response = requests.get(SERVER_FEEDBACK_URL, timeout=3)
            response.raise_for_status()
            data = response.json()
            with feedback_lock:
                feedback_text = data.get('feedback', 'No feedback')
        except Exception as e:
            with feedback_lock:
                feedback_text = "Error fetching feedback"
        time.sleep(0.5)

"""
Similar to 'fetch_feedback(). It will retrieve the annotated 
image from the AWS server

Uses the requests libray to fetch the data to get and download 
the image. Also resizes the image to be 640 x 480, in case 
the annotated image exceeded this
"""

def fetch_pose_image():
    global pose_prediction_image
    while True:
        try:
            response = requests.get(SERVER_POSE_IMAGE_URL, timeout=3)
            response.raise_for_status()
            img_array = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
            img_resized = cv2.resize(img_array, (640, 480))
            with pose_lock:
                pose_prediction_image = img_resized
        except Exception as e:
            with pose_lock:
                pose_prediction_image = np.zeros((480, 640, 3), dtype=np.uint8)
        time.sleep(1.0)  

"""
img1 and img2 are both frames 

Function that will compare the last two images from the webcam
to see how similar they are. Returns an int. If this int is too small,
the images are almost identical so don't upload the image
"""
def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

"""
Function that will add the feedback to the live feed

It will place it at the top left of the image and 
also put it inside a black rectangle for better visibility
"""
def overlay_text(frame, text):
    overlay_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10
    text_y = 30

    cv2.rectangle(overlay_frame, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

    cv2.putText(overlay_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return overlay_frame

"""
Main function

Initializes the webcam, the threads, and calls all other functions
to capture images, compare them, send them, and then display them
"""
def main():
    capture_screenshot(first_frame=True)

    threading.Thread(target=fetch_feedback, daemon=True).start()
    threading.Thread(target=fetch_pose_image, daemon=True).start()

    while True:
        if not capture_screenshot(first_frame=False):
            break

        img_current = cv2.imread(IMAGE_PATH)
        img_last = cv2.imread(LAST_IMAGE_PATH)

        if img_current is None or img_last is None:
            continue

        img_current_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
        img_last_gray = cv2.cvtColor(img_last, cv2.COLOR_BGR2GRAY)

        error = mse(img_current_gray, img_last_gray)

        if error > 150:
            threading.Thread(target=upload_image, daemon=True).start()
            
        with feedback_lock:
            feedback = feedback_text

        display_frame = overlay_text(img_current, feedback)

        with pose_lock:
            pose_img_copy = pose_prediction_image.copy()

        combined = np.hstack((display_frame, pose_img_copy))

        cv2.imshow("Pose Helper + Prediction", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.2)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
