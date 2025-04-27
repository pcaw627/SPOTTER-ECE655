from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt
import json
import requests
import time
import datetime
import random
import math


'''
Converts the raw landmark coords into a dictionary, mapping a string of the body part to its detected location.
'''
def get_body_part_map(pose_landmarks_proto):
    # Define the mapping of indices to body part names
    body_parts = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
        29: "left_heel",
        30: "right_heel",
        31: "left_foot_index",
        32: "right_foot_index"
    }
    
    bodymap = {}
    
    for i, landmark in enumerate(pose_landmarks_proto.landmark):
        if i in body_parts:
            body_part_name = body_parts[i]
            bodymap[body_part_name] = (landmark.x, landmark.y, landmark.z)
    
    return bodymap


'''
Annotates an image with landmarks, and returns a map mapping body parts to their coords. 
'''
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    pose_body_maps = []

    # Loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    
        # print(pose_landmarks_proto)
        body_map = get_body_part_map(pose_landmarks_proto)
        pose_body_maps.append(body_map)

    return annotated_image, pose_body_maps

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    # , delegate=python.BaseOptions.Delegate.GPU)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    output_segmentation_masks=False,
    min_pose_detection_confidence = 0.3,
    min_pose_presence_confidence = 0.3,
    min_tracking_confidence = 0.3)
detector = vision.PoseLandmarker.create_from_options(options)

def pose_detect(filename="image.jpg"):
    # Test image.
    img = cv2.imread(filename)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(filename)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    annotated_image, body_maps = draw_landmarks_on_image(image.numpy_view()[:,:,::-1], detection_result)
    # plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    plt.imsave("pose_prediction.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # print(json.dumps(body_maps[0], indent=2))
    return body_maps

def download_screenshot():
    print(f"Downloading screenshot at {time.strftime('%H:%M:%S')}")
    try:
        # response = requests.get("http://3.15.203.82/files/screenshot.png")
        response = requests.get("http://3.15.203.82/files/image.jpg")
        if response.status_code == 200:
            # with open("screenshot.png", "wb") as f:
            with open("image.jpg", "wb") as f:
                f.write(response.content)
            
            print("Download successful, running pose analysis")

        else:
            print(f"Failed to download: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading: {e}")

def update_prediction():
    start_time = time.perf_counter()

    download_screenshot()
    # pose_detect(filename="screenshot.png")
    body_maps = pose_detect()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Downloaded and predicted in {elapsed_time:.4f} seconds")
    return body_maps

def classify_exercise(bodymap):
    """
    Classify exercise as either pushup or squat based on pose landmarks
    
    Args:
        bodymap: Dictionary mapping body parts to their coordinates
    
    Returns:
        String: "pushup" or "squat" or "neutral"
    """
    
    wrists_mid = np.mean([
        bodymap["left_wrist"][:2], 
        bodymap["right_wrist"][:2]
    ], axis=0)
    
    ankles_mid = np.mean([
        bodymap["left_ankle"][:2], 
        bodymap["right_ankle"][:2]
    ], axis=0)

    ankle_left = bodymap["left_ankle"][:2]
    ankle_right = bodymap["right_ankle"][:2]
    wrist_left = bodymap["left_wrist"][:2]
    wrist_right = bodymap["right_wrist"][:2]
    
    # this classification is just looking at whether the y coord of the wrists are on relatively the same level as the ankles. 
    # NOTE: we are dealing with 3D data, so this will break if its a head on view (where the slope will be near vertical.)
    # as a quick (but not complete fix to this, we can instead take the slope between right wrist / left ankle, left wrist / right ankle.)
    # dy = wrists[1] - ankles[1]
    # dx = wrists[0] - ankles[0]
    # wrist_foot_slope = abs(dy/dx)
    # is_horizontal = wrist_foot_slope <= 1

    # this should work better, regardless of z-orientation.
    dy1 = wrist_left[1] - ankle_right[1]
    dx1 = wrist_left[0] - ankle_right[0]
    leftwrist_rightfoot_slope = abs(dy1/dx1)
    
    dy2 = wrist_right[1] - ankle_left[1]
    dx2 = wrist_right[0] - ankle_left[0]
    rightwrist_leftfoot_slope = abs(dy2/dx2)

    # if both slopes are less than 45 degrees
    wrist_pair = (wrist_left, wrist_right)
    ankle_pair = (ankle_left, ankle_right)
    wrist_ankle_slope_is_horizontal = keypoint_pairs_are_horizontal(wrist_pair, ankle_pair, 45)
    # wrist_ankle_slope_is_horizontal = (leftwrist_rightfoot_slope <= 1) & (rightwrist_leftfoot_slope <= 1)
    
    
    if wrist_ankle_slope_is_horizontal:
        return "pushup"
    else:
        return "squat"
    
def upload_json_response(exercise_feedback, pose_classification):
    timestamp = str(datetime.datetime.now())
    feedback = exercise_feedback
    feedback_file = exercise_feedback + ".wav"

    data = {
        'timestamp': timestamp, 
        'feedback': feedback, 
        'pose_classification': pose_classification, 
        'feedback_audio_file': feedback_file
    }

    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.post(
        url='http://3.15.203.82/json',
        data=json.dumps(data),
        headers=headers
    )

    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
def upload_test_image():
    workout_imgs = [
        "test_imgs\pushup_DOWN_dip.jpg",
        "test_imgs\pushup_DOWN_flat.jpg",
        "test_imgs\pushup_DOWN_hump.jpg",
        "test_imgs\pushup_UP_dip.jpg",
        "test_imgs\pushup_UP_flat.jpg",
        "test_imgs\pushup_UP_flat_2.jpg",
        "test_imgs\pushup_UP_hump.jpg",
        "test_imgs\pushupforward_DOWN_dip.jpg",
        "test_imgs\pushupforward_DOWN_flat.jpg",
        "test_imgs\pushupforward_DOWN_hump.jpg",
        "test_imgs\pushupforward_UP_dip.jpg",
        "test_imgs\pushupforward_UP_flat.jpg",
        "test_imgs\pushupforward_UP_hump.jpg",

        "test_imgs\squat_diag_legs_close_DOWN.jpg",
        "test_imgs\squat_diag_legs_close_UP.jpg",
        "test_imgs\squat_diag_legs_spread_DOWN.jpg",
        "test_imgs\squat_diag_legs_spread_UP.jpg",
        "test_imgs\squat_diag_legs_spread_UP_2.jpg",
        "test_imgs\squat_forward_DOWN.jpg",
        "test_imgs\squat_forward_legs_close_MID.jpg",
        "test_imgs\squat_forward_legs_spread_MID.jpg",
        "test_imgs\squat_forward_UP.jpg",
        "test_imgs\squat_side_legs_close_MID.jpg",
        "test_imgs\squat_side_legs_close_MID_2.jpg",
        "test_imgs\squat_side_legs_close_UP.jpg",
        "test_imgs\squat_side_legs_close_UP_2.jpg"
    ]

    image_path = random.choice(workout_imgs)

    with open(image_path, 'rb') as image_file:
        files = {'file': ('image.jpg', image_file)} 
        response = requests.post('http://3.15.203.82/upload', files=files)   

        if response.status_code == 200:
            print(f"Image {image_path} uploaded successfully as 'image.jpg'!")

        else:
            print(f"Failed to upload {image_path}. Status code: {response.status_code}")
    timestamp = str(datetime.datetime.now())
    feedback = exercise_feedback
    feedback_file = exercise_feedback + ".wav"

    data = {
        'timestamp': timestamp, 
        'feedback': feedback, 
        'pose_classification': pose_classification, 
        'feedback_audio_file': feedback_file
    }

    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.post(
        url='http://3.15.203.82/json',
        data=json.dumps(data),
        headers=headers
    )

    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
def upload_annotated_image():
    image_path = "pose_prediction.jpg"

    with open(image_path, 'rb') as image_file:
        files = {'file': ('pose_prediction.jpg', image_file)} 
        response = requests.post('http://3.15.203.82/upload', files=files)   

        if response.status_code == 200:
            print(f"Image {image_path} uploaded successfully as 'pose_prediction.jpg'!")

        else:
            print(f"Failed to upload {image_path}. Status code: {response.status_code}")

# Takes three keypoints (not necessarily connected), and returns the angle between them (taking keypoint_joint as the "hinge" of the angle).
def keypoint_angle(keypoint_start, keypoint_joint, keypoint_end):    
    # Calculate vectors from joint to start and end points
    x1, y1 = keypoint_start[0] - keypoint_joint[0], keypoint_start[1] - keypoint_joint[1]
    x2, y2 = keypoint_end[0] - keypoint_joint[0], keypoint_end[1] - keypoint_joint[1]
    
    # Calculate the angle using arctan2
    angle1 = math.atan2(y1, x1)
    angle2 = math.atan2(y2, x2)
    
    # Find the difference between the angles
    angle = angle2 - angle1
    
    # Convert to degrees and normalize to range [0, 360)
    angle_deg = math.degrees(angle)
    angle_deg = angle_deg % 360
    
    return angle_deg

# Takes two keypoints (not necessarily connected), and returns the slope between them (taking the horizontal as 0 degrees).
def keypoint_slope(keypoint_start, keypoint_end):
    dy = keypoint_end[1] - keypoint_start[1]
    dx = keypoint_end[0] - keypoint_start[0]

    return dy/dx

"""
Takes two keypoints (not necessarily connected), and returns a NAIVE estimate of whether they are positioned horizontally (taking the horizontal as 0 degrees).
Returns true if the slope is less than the threshold slope angle alpha.
"""
def keypoints_are_horizontal(keypoint_start, keypoint_end, alpha):
    slope = keypoint_slope(keypoint_start, keypoint_end)
    slope_alpha = abs(np.atan(alpha))

    return (abs(slope) < slope_alpha)

"""
Takes two *pairs* of keypoints (not necessarily connected), and returns the slope between them (taking the horizontal as 0 degrees). 
+ The points within each tuple must be associated with each other (for example, left-shoulder & right-shoulder).
+ The points in each tuple should be ordered in the same manner (left hand, right hand) & (left foot, right foot)
If you want to use this method with a single keypoint and a pair, then you can pass the single keypoint as (single_keypoint, single_keypoint).

This method is more reliable for finding slope when the endpoints are seen from a head-on angle, which due to perspective can appear as vertical instead.
Returns true if the slope is less than the threshold slope angle +/- alpha.
"""
def keypoint_pairs_are_horizontal(keypoint_start_pair, keypoint_end_pair, alpha):
    slope_1 = keypoint_slope(keypoint_start_pair[0], keypoint_end_pair[1])
    slope_2 = keypoint_slope(keypoint_start_pair[1], keypoint_end_pair[0])
    slope_alpha = abs(np.atan(alpha))

    return (abs(slope_1) < slope_alpha) & (abs(slope_2) < slope_alpha)

"""
Returns true if the joint is above the line drawn from start to end points.
"""
def is_above_line(keypoint_start, keypoint_end, keypoint_joint):
    # first interpolate where the joint's y-coord would be if it was perfectly in line from start to end. 
    # let's call this y_joint*. 
    y_joint_star = interpolate(keypoint_start, keypoint_end, x_sample=keypoint_joint[0])

    return keypoint_joint[1] > y_joint_star

# Returns the y-coord of an x-coord sample if taken along the line from start to end point.
def interpolate(keypoint_start, keypoint_end, x_sample):
    x1, y1 = keypoint_start[0], keypoint_start[1]
    x2, y2 = keypoint_end[0], keypoint_end[1]

    slope = (y2-y1) / (x2-x1)
    b = y1 - slope * x1
    y_sample = slope * x_sample + b

    return y_sample

def get_feedback(pose_classification, pose):
    
    ankle_right = pose["right_ankle"][:2]
    ankle_left  = pose["left_ankle"][:2]
    wrist_right = pose["right_wrist"][:2]
    wrist_left  = pose["left_wrist"][:2]
    shoulder_right = pose["right_shoulder"][:2]
    shoulder_left  = pose["left_shoulder"][:2]
    hip_right = pose["right_hip"][:2]
    hip_left  = pose["left_hip"][:2]
    elbow_right = pose["right_elbow"][:2]
    elbow_left  = pose["left_elbow"][:2]
    knee_right = pose["right_knee"][:2]
    knee_left  = pose["left_knee"][:2]
    lip_right = pose["right_lip"][:2]
    lip_left  = pose["left_lip"][:2]

    wrists_mid = np.mean([bodymap["left_wrist"][:2], bodymap["right_wrist"][:2]], axis=0)
    shoulders_mid = np.mean([bodymap["left_shoulder"][:2], bodymap["right_shoulder"][:2]], axis=0)
    ankles_mid = np.mean([bodymap["left_ankle"][:2], bodymap["right_ankle"][:2]], axis=0)
    hips_mid = np.mean([bodymap["left_hip"][:2], bodymap["right_hip"][:2]], axis=0)
    knees_mid = np.mean([bodymap["left_knee"][:2], bodymap["right_knee"][:2]], axis=0)
    elbows_mid = np.mean([bodymap["left_elbow"][:2], bodymap["right_elbow"][:2]], axis=0)
    lips_mid = np.mean([bodymap["left_lip"][:2], bodymap["right_lip"][:2]], axis=0)

    feedback = []
    
    if pose_classification == "pushup":

        # bad form, take care of worst issues first
        shoulders_knees_hips_angle = keypoint_angle(shoulders_mid, hips_mid, keypoint_joint=knees_mid)
        # if back is arched up (shoulder-hip-knee angle is deviates from 180 degrees by >20degrees , hip lies ABOVE line from shoulder to knees)
        if (shoulders_knees_hips_angle > 20 \
            & is_above_line(shoulders_mid, knees_mid, keypoint_joint=hips_mid)):
            feedback.append("lower your hips!")
    
        # if back is arched down (shoulder-hip-knee angle is deviates from 180 degrees, hip lies BELOW line from shoulder to knees)
        if (shoulders_knees_hips_angle > 20 \
            & ~is_above_line(shoulders_mid, knees_mid, keypoint_joint=hips_mid)):
            feedback.append("raise your hips, don't slouch")
    
        # no need for this one (and lots of potential for error if observing directly from the side)
        # return "put your arms at shoulder width apart"

        # if elbows aren't fully straightened
        if (keypoint_angle(shoulders_mid, wrists_mid, keypoint_joint=elbows_mid) > 20 \
            & is_above_line(shoulders_mid, wrists_mid, keypoint_joint=elbows_mid)):
            feedback.append("fully extend!")
    
        # return "don't flare elbow, tuck them in!"
    
        # if slope from midpoint of lip markers to tip of nose deviates from the slope of back (midpoint of hips to midpoint of shoulders)
        lip_nose_slope = keypoint_slope(lips_mid, pose["nose"][:2])
        back_slope = keypoint_slope(hips_mid, shoulders_mid)
        if (abs(lip_nose_slope - back_slope) > 40):
            feedback.append("don't look up!")

        # good
        praise = ["perfect, keep going!",
                  "nice form, make sure to keep your core engaged!"]
        
        # only praise if we didn't have any negative feedback
        if len(feedback) == 0:
            feedback.append(random.choice(praise))
        
        return feedback
        
    if pose_classification == "squat":
        
        # bad form, take care of worst issues first

        # if slope of hips to knees deviates from horizontal
        if (keypoint_slope(hips_mid, knees_mid) > 30):
            return "squat deeper!"
    
        # if slope from back deviates from vertical
        return "keep your back straight!"
    
        # if slope from back deviates from vertical more than 40 degrees
        return "don't lean forward too much!"
    
        # if slope of heel-toe deviates from vertical
        return "don't  lift your heels from the ground!"
    
        # if angle of toe-heel-knee deviates too much from 90 degrees
        return "keep your knees behind your toes!"
    
        # if neck angle is more than 45 degrees
        return "no turtle neck!"
    

        # good
        praise = ["perfect, keep going!",
                  "nice form, make sure to hinge at the hips!",
                  "great form, push up through your heels! "]
        
        # only praise if we didn't have any negative feedback
        if len(feedback) == 0:
            feedback.append(random.choice(praise))
        
        return feedback

        
    if pose_classification == "neutral":
        return "let's get to it!"


if __name__ == '__main__':
    print("Beginning server")
    # download_screenshot()

    # update_prediction()

    # interval_seconds = 60  # Change this to your desired interval
    # schedule.every(interval_seconds).seconds.do(update_prediction)

    while True:
        
        # upload dummy data
        upload_test_image()
        time.sleep(1)

        # pull stored data on AWS and predict
        body_maps = update_prediction()

        # classify exercise
        pose = body_maps[0] # get just one pose of interest
        pose_classification = classify_exercise(pose)
        print(pose_classification)
        time.sleep(1)

        feedback = get_feedback(pose_classification, pose)
        feedback = "DUMMY DATA - you're doing great!"

        # send json response to AWS with classification and feedback
        upload_json_response(feedback, pose_classification)

        time.sleep(2)
        # if pose_classification == 'squat':
        #     break



