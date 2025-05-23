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
from datetime import datetime
import random
import math
import os

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    # , delegate=python.BaseOptions.Delegate.GPU)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    output_segmentation_masks=False,
    min_pose_detection_confidence = 0.4,
    min_pose_presence_confidence = 0.3,
    min_tracking_confidence = 0.3)
detector = vision.PoseLandmarker.create_from_options(options)


PUSHUP_THRESHOLD = 145
SQUAT_THRESHOLD = 135


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

'''
Annotates an image with highlights, using a list of segment tuples of the desired body parts. Also overlays text (in the top left by default).
'''
def highlight_body_segments(pose, segments_to_highlight, text_message=None):
    
    filename = "pose_prediction.jpg"
    image = cv2.imread(filename)
    highlighted_image = np.copy(image)
    h, w = highlighted_image.shape[:2]
    
    # draw each segment to highlight
    for start_part, end_part in segments_to_highlight:
        if start_part in pose and end_part in pose:
            start_point = pose[start_part]
            end_point = pose[end_part]
            
            # change normalized coordinates to pixel coordinates
            start_pixel = (int(start_point[0] * w), int(start_point[1] * h))
            end_pixel = (int(end_point[0] * w), int(end_point[1] * h))
            
            cv2.line(highlighted_image, 
                     start_pixel, 
                     end_pixel, 
                     color=(0, 0, 255),  # bgr format: red
                     thickness=5)
            
    if text_message:
        text_position = (20, 40) # top left
        lines = text_message.split('\n')
        
        # background
        line_height = 30  # height per line
        max_width = 0
        
        # find the widest line to size our background
        for line in lines:
            width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            max_width = max(max_width, width)
        
        cv2.rectangle(highlighted_image, 
                     (text_position[0] - 10, text_position[1] - 30), 
                     (text_position[0] + max_width + 10, text_position[1] + (len(lines) - 1) * line_height + 10), 
                     (255, 255, 255), 
                     -1)  # Filled rectangle
        
        # text overlay
        for i, line in enumerate(lines):
            y_position = text_position[1] + i * line_height
            cv2.putText(highlighted_image, 
                       line, 
                       (text_position[0], y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       (0, 0, 0), #black
                       2)  # thickness

    plt.imsave("highlighted.jpg", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
    
    # return np array of highlighted image
    return highlighted_image

'''
Performs pose estimation on the given file, returning a maps of all bodies in shot (up to a limit of num_poses, set in options at top.)
'''
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
    plt.imsave("pose_prediction.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # print(json.dumps(body_maps[0], indent=2))
    return body_maps

"""
Attempts to download and save an image from AWS server.
"""
def download_screenshot():
    print(f"Downloading screenshot at {time.strftime('%H:%M:%S')}")
    try:
        response = requests.get("http://3.15.203.82/files/screenshot.jpg")
        # response = requests.get("http://3.15.203.82/files/image.jpg")
        if response.status_code == 200:
            with open("screenshot.jpg", "wb") as f:
            # with open("image.jpg", "wb") as f:
                f.write(response.content)
            
            print("Download successful, running pose analysis")

        else:
            print(f"Failed to download: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading: {e}")

"""
Predicts poses, returning a maps of all bodies in shot (up to a limit of num_poses, set in options at top.) Prints time of execution.
If testing, pulls locally; otherwise attempts to download from AWS. 
"""
def update_prediction(TESTING=False, filename="screenshot.jpg"):
    start_time = time.perf_counter()
    body_maps = {}

    if TESTING:
        print("FILE: ", filename)
        cv2.imwrite("screenshot.jpg", cv2.imread(filename))
    else:
        download_screenshot()
    
    body_maps = pose_detect(filename="screenshot.jpg")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if TESTING:
        print(f"Retrieved local image and predicted in {elapsed_time:.4f} seconds")
    else:
        print(f"Downloaded image and predicted in {elapsed_time:.4f} seconds")

    return body_maps

"""
Classifies pose into various exercises. 
Currently we have pushups and squats (with an UP and DOWN for each) but this can be expanded to any number of exercises.
"""
def classify_exercise(pose):
    
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
    lip_right = pose["mouth_right"][:2]
    lip_left  = pose["mouth_left"][:2]
    nose  = pose["nose"][:2]

    wrists_mid = np.mean([pose["left_wrist"][:2], pose["right_wrist"][:2]], axis=0)
    shoulders_mid = np.mean([pose["left_shoulder"][:2], pose["right_shoulder"][:2]], axis=0)
    ankles_mid = np.mean([pose["left_ankle"][:2], pose["right_ankle"][:2]], axis=0)
    hips_mid = np.mean([pose["left_hip"][:2], pose["right_hip"][:2]], axis=0)
    knees_mid = np.mean([pose["left_knee"][:2], pose["right_knee"][:2]], axis=0)
    elbows_mid = np.mean([pose["left_elbow"][:2], pose["right_elbow"][:2]], axis=0)
    lips_mid = np.mean([pose["mouth_left"][:2], pose["mouth_right"][:2]], axis=0)

    shoulders_knees_hips_angle = keypoint_angle(shoulders_mid, knees_mid, hips_mid)
    hips_are_above_shoulders = is_above_line(shoulders_mid, hips_mid, knees_mid)
    left_elbow_angle = keypoint_angle(shoulder_left, elbow_left, wrist_left)
    right_elbow_angle = keypoint_angle(shoulder_right, elbow_right, wrist_right)
    left_knee_angle = keypoint_angle(hip_left, knee_left, ankle_left)
    right_knee_angle = keypoint_angle(hip_right, knee_right, ankle_right)
    lip_nose_slope = keypoint_slope_degrees(lips_mid, nose)
    back_slope = keypoint_slope_degrees(hips_mid, shoulders_mid)
    thigh_slope = keypoint_slope_degrees(hips_mid, knees_mid)
    
    # this classification is just looking at whether the y coord of the wrists are on relatively the same level as the ankles. 
    # NOTE: we are dealing with 3D data, so this will break if its a head on view (where the slope will be near vertical.)
    # as a quick (but not complete) fix to this, we can instead take the slope between right wrist / left ankle, left wrist / right ankle.

    # if both slopes are less than 45 degrees
    wrist_pair = (wrist_left, wrist_right)
    ankle_pair = (ankle_left, ankle_right)
    # this function uses the approach in note above
    # wrist_ankle_slope_is_horizontal = keypoint_pairs_are_horizontal(wrist_pair, ankle_pair, 45)
    wrist_ankle_slope_is_horizontal = abs(keypoint_slope_degrees(ankle_left, wrist_left)) < 45 and abs(keypoint_slope_degrees(ankle_right, wrist_right) < 45)
    

    print("joint angles", left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle)

    if wrist_ankle_slope_is_horizontal:
        if abs(left_elbow_angle) > PUSHUP_THRESHOLD and abs(right_elbow_angle) > PUSHUP_THRESHOLD:
            return "pushup_UP"
        else: 
            return "pushup_DOWN"
    else:
        if abs(left_knee_angle) > SQUAT_THRESHOLD and abs(right_knee_angle) > SQUAT_THRESHOLD:
            return "neutral" # which is the same as squat_UP
        else:
            return "squat_DOWN"
        
    return "neutral"

"""
Uploads feedback and classification in a text response to AWS server in JSON format. 
"""
def upload_json_response(exercise_feedback, pose_classification):
    timestamp = str(datetime.now())
    feedback = exercise_feedback
    feedback_file = exercise_feedback + ".mp3"

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

    
    # print(f"Status Code: {response.status_code}")
    # print(f"Response: {response.text}")

"""
Uploads a random test image to AWS.
"""
def upload_test_image(verbose=False):
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

        if verbose:
            if response.status_code == 200:
                print(f"Image {image_path} uploaded successfully as 'image.jpg'!")

            else:
                print(f"Failed to upload {image_path}. Status code: {response.status_code}")
    
    return image_path

"""
Uploads pose_prediction.png to AWS.
"""
# TODO: rewrite to change filename
def upload_annotated_image(verbose=False, image_path="screenshot.jpg"):
    # image_path = "pose_prediction.png"

    with open(image_path, 'rb') as image_file:
        files = {'file': ('pose_prediction.jpg', image_file)} 
        response = requests.post('http://3.15.203.82/upload', files=files)   

        if verbose:
            if response.status_code == 200:
                print(f"Image {image_path} uploaded successfully as '{image_path}'!")

            else:
                print(f"Failed to upload {image_path}. Status code: {response.status_code}")

"""
Takes three keypoints (not necessarily connected), and returns the angle between them (taking keypoint_joint as the "hinge" of the angle).
Returns value between [-180,180).
"""
def keypoint_angle(start, joint, end):    
    # Calculate vectors from joint to start and end points
    x1, y1 = start[0] - joint[0], start[1] - joint[1]
    x2, y2 = end[0] - joint[0], end[1] - joint[1]
    
    # Calculate the angle using arctan2
    angle1 = math.atan2(y1, x1)
    angle2 = math.atan2(y2, x2)
    
    # Find the difference between the angles
    angle = angle2 - angle1
    
    # Convert to degrees and normalize to range [0, 360)
    angle_deg = math.degrees(angle)
    
    return ((angle_deg + 180) % 360) - 180

"""
Takes two keypoints (not necessarily connected), and returns the slope between them (taking the horizontal as 0 degrees). Returns the numerical slope (dy/dx).
"""
def keypoint_slope(start, end):
    dy = end[1] - start[1]
    dx = end[0] - start[0]

    return dy/dx

"""
Takes two keypoints (not necessarily connected), and returns the slope between them (taking the horizontal as 0 degrees). Returns the slope in degrees.
"""
def keypoint_slope_degrees(start, end):
    return keypoint_angle(start=(start[0]+1, start[1]), joint=start, end=end)

"""
Takes two keypoints (not necessarily connected), and returns a NAIVE estimate of whether they are positioned horizontally (taking the horizontal as 0 degrees).
Returns true if the slope is less than the threshold slope angle alpha.
"""
def keypoints_are_horizontal(start, end, alpha):
    slope = keypoint_slope_degrees(start, end)
    # slope_alpha = abs(math.atan2(alpha))

    return (abs(slope) < alpha)

"""
Takes two *pairs* of keypoints (not necessarily connected), and returns the slope between them (taking the horizontal as 0 degrees). 
+ The points within each tuple must be associated with each other (for example, left-shoulder & right-shoulder).
+ The points in each tuple should be ordered in the same manner (left hand, right hand) & (left foot, right foot)
If you want to use this method with a single keypoint and a pair, then you can pass the single keypoint as (single_keypoint, single_keypoint).

This method is more reliable for finding slope when the endpoints are seen from a head-on angle, which due to perspective can appear as vertical instead.
Returns true if the slope is less than the threshold slope angle +/- alpha.
"""
def keypoint_pairs_are_horizontal(start_pair, end_pair, alpha):
    slope_1 = keypoint_slope_degrees(start_pair[0], end_pair[1])
    slope_2 = keypoint_slope_degrees(start_pair[1], end_pair[0])
    # slope_alpha = abs(math.atan2(alpha))

    return (abs(slope_1) < alpha) & (abs(slope_2) < alpha)

"""
Returns true if the joint is above the line drawn from start to end points.
"""
def is_above_line(start, end, joint):
    # first interpolate where the joint's y-coord would be if it was perfectly in line from start to end. 
    # let's call this y_joint*. 
    y_joint_star = interpolate(start, end, x_sample=joint[0])

    return joint[1] > y_joint_star

"""
Returns the y-coord of an x-coord sample if taken along the line from start to end point.
"""
def interpolate(start, end, x_sample):
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]

    slope = (y2-y1) / (x2-x1)
    b = y1 - slope * x1
    y_sample = slope * x_sample + b

    return y_sample

"""
Analyzes pose landmarks to determine what feedback to forward to user for proper form.
"""
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
    lip_right = pose["mouth_right"][:2]
    lip_left  = pose["mouth_left"][:2]
    nose  = pose["nose"][:2]

    wrists_mid = np.mean([pose["left_wrist"][:2], pose["right_wrist"][:2]], axis=0)
    shoulders_mid = np.mean([pose["left_shoulder"][:2], pose["right_shoulder"][:2]], axis=0)
    ankles_mid = np.mean([pose["left_ankle"][:2], pose["right_ankle"][:2]], axis=0)
    hips_mid = np.mean([pose["left_hip"][:2], pose["right_hip"][:2]], axis=0)
    knees_mid = np.mean([pose["left_knee"][:2], pose["right_knee"][:2]], axis=0)
    elbows_mid = np.mean([pose["left_elbow"][:2], pose["right_elbow"][:2]], axis=0)
    lips_mid = np.mean([pose["mouth_left"][:2], pose["mouth_right"][:2]], axis=0)

    shoulders_knees_hips_angle = keypoint_angle(shoulders_mid, hips_mid, knees_mid)
    hips_are_above_shoulders = is_above_line(shoulders_mid, hips_mid, knees_mid)
    left_elbow_angle = keypoint_angle(shoulder_left, elbow_left, wrist_left)
    right_elbow_angle = keypoint_angle(shoulder_right, elbow_right, wrist_right)
    lip_nose_slope = keypoint_slope_degrees(lips_mid, nose)
    back_slope = keypoint_slope_degrees(hips_mid, shoulders_mid)
    thigh_slope = keypoint_slope_degrees(hips_mid, knees_mid)

    feedback = []
    problem_limbs = set()
    
    if pose_classification == "pushup_DOWN":
        # bad form, take care of worst issues first
        # if back is arched up (shoulder-hip-knee angle is deviates from 180 degrees by >20degrees , hip lies ABOVE line from shoulder to knees)
        if (abs(shoulders_knees_hips_angle) > 20 and hips_are_above_shoulders):
            feedback.append("lower your hips!")
            problem_limbs.update({('right_hip', 'right_shoulder'), ('right_hip', 'right_knee'), ('left_hip', 'left_shoulder'), ('left_hip', 'left_knee')})
    
        # if back is arched down (shoulder-hip-knee angle is deviates from 180 degrees, hip lies BELOW line from shoulder to knees)
        if (abs(shoulders_knees_hips_angle) > 20 and not hips_are_above_shoulders):
            feedback.append("raise your hips, don't slouch")
            problem_limbs.update({('right_hip', 'right_shoulder'), ('right_hip', 'right_knee'), ('left_hip', 'left_shoulder'), ('left_hip', 'left_knee')})
    
        # no need for this one (and lots of potential for error if observing directly from the side)
        # return "put your arms at shoulder width apart"

        # return "don't flare elbow, tuck them in!"
    
        # if slope from midpoint of lip markers to tip of nose deviates from the slope of back (midpoint of hips to midpoint of shoulders)
        if (abs(lip_nose_slope - back_slope) > 40):
            feedback.append("don't look up!")
            problem_limbs.update({('right_eye_inner', 'nose'), ('left_eye_inner', 'nose')})

        # good
        praise = ["chiseled abs! chiseled abs!", "nice form, make sure to keep your core engaged!", "hydrated?", "champion!"]
        
        # only praise if we didn't have any negative feedback
        if len(feedback) == 0:
            feedback.append(random.choice(praise))

    
    if pose_classification == "pushup_UP":
        # bad form, take care of worst issues first
        # if back is arched up (shoulder-hip-knee angle is deviates from 180 degrees by >20degrees , hip lies ABOVE line from shoulder to knees)
        if (abs(shoulders_knees_hips_angle) > 20 and hips_are_above_shoulders):
            feedback.append("lower your hips!")
            problem_limbs.update({('right_hip', 'right_shoulder'), ('right_hip', 'right_knee'), ('left_hip', 'left_shoulder'), ('left_hip', 'left_knee')})
    
        # if back is arched down (shoulder-hip-knee angle is deviates from 180 degrees, hip lies BELOW line from shoulder to knees)
        if (abs(shoulders_knees_hips_angle) > 20 and not hips_are_above_shoulders):
            feedback.append("raise your hips, don't slouch")
            problem_limbs.update({('right_hip', 'right_shoulder'), ('right_hip', 'right_knee'), ('left_hip', 'left_shoulder'), ('left_hip', 'left_knee')})

        # if elbows aren't fully straightened (need to measure each side independently)
        if (abs(left_elbow_angle) < 160 and abs(right_elbow_angle) < 160):
            feedback.append("fully extend your elbows!")
            problem_limbs.update({('right_elbow', 'right_shoulder'), ('right_elbow', 'right_wrist'), ('left_elbow', 'left_shoulder'), ('left_elbow', 'left_wrist')})

        # good
        praise = ["And... down!", "down!", "next rep!", "hydrated?", "champion!"]
        
        if len(feedback) == 0:
            feedback.append(random.choice(praise))
        

    if pose_classification == "squat_DOWN":
        # bad form, take care of worst issues first

        # if slope of hips to knees deviates from horizontal
        if (abs(thigh_slope) > 30):
            feedback.append("squat deeper!")
            problem_limbs.update({('right_hip', 'right_knee'), ('left_hip', 'left_knee')})
    
        # if slope from back deviates from vertical more than 40 degrees
        if (abs(90-back_slope) > 50):
            straighten_back_feedback = ["don't lean forward too much!", "keep your back straight!"]
            feedback.append(random.choice(straighten_back_feedback))
            problem_limbs.update({('right_hip', 'right_shoulder'), ('left_hip', 'left_shoulder')})
    
        # if slope of heel-toe deviates from vertical
        # return "don't  lift your heels from the ground!"
    
        # if angle of toe-heel-knee deviates too much from 90 degrees
        # return "keep your knees behind your toes!"
    
        # if neck angle is more than 45 degrees
        # return "no turtle neck!"
    

        # good
        praise = ["perfect, keep going!",
                  "nice form, make sure to hinge at the hips!",
                  "great form, push up through your heels! ",
                  "hydrated?",
                  "champion!"]
        
        # only praise if we didn't have any negative feedback
        if len(feedback) == 0:
            feedback.append(random.choice(praise))
        

    if pose_classification == "neutral":
        feedback.append("let's get to it!")

    print("Feedback: ", feedback)
    # print("HIGHLIGHTED LIMBS: ", problem_limbs)
    return feedback, problem_limbs
        


if __name__ == '__main__':
    print("Beginning server")
    download_screenshot()
    update_prediction()

    TESTING = False
    TEST_IMAGE = ""

    test_path = 'test_imgs_out'
    test_file_count = sum(1 for item in os.scandir(test_path) if item.is_file())
    body_maps = {}

    while True:
    # for sample in range(20):
        process_start_time = datetime.now()
        
        if TESTING:
            # upload dummy data
            TEST_IMAGE = upload_test_image()
            # time.sleep(1)

        # pull stored data on AWS and predict
        if TESTING:
            body_maps = update_prediction(TESTING=TESTING, filename=TEST_IMAGE)
        else:
            body_maps = update_prediction()

        # classify exercise
        if (len(body_maps)):
            pose = body_maps[0] # get just one pose of interest
            pose_classification = classify_exercise(pose)
            print(pose_classification)
            # time.sleep(1)

            feedback, problem_limbs = get_feedback(pose_classification, pose)
            # feedback = "DUMMY DATA - you're doing great!"

            # send json response to AWS with classification and feedback
            upload_json_response(feedback[0], pose_classification)
            # upload_annotated_image(image_path="pose_prediction.png")
            upload_annotated_image(image_path="highlighted.jpg")

            feedback_concat = pose_classification+"\n"+"\n".join(feedback)
            highlighted_image = highlight_body_segments(pose, problem_limbs, feedback_concat)

        else:
            print("No body detected in frame.")

        
        process_time = datetime.now() - process_start_time
        print("PROCESS TIME: ", process_time)
        
        # image = cv2.imread("highlighted.png")
        # test_copy = np.copy(image)
        # plt.imsave("test_imgs_out/test_"+str(test_file_count+sample)+".png", cv2.cvtColor(test_copy, cv2.COLOR_RGB2BGR))

        time.sleep(0.25)
        # if pose_classification == 'squat':
        #     break

        print("="*60, "\n")



