import cv2
import mediapipe as mp
import numpy as np

def draw_rectangle(event, x, y, flags, params):
    global rectangles, drawing, current_rectangle, current_rectangle_id

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_rectangle_id += 1
        current_rectangle = [(x, y), (x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rectangle[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles[current_rectangle_id] = current_rectangle
        print(f"Coordinates of the rectangle: ({rectangles}")
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     print(x,y)

# Define the callback function for deleting rectangles
def delete_rectangle(event, x, y, flags, params):
    global rectangles
    if event == cv2.EVENT_LBUTTONDOWN:
        for rect in rectangles:
            if rect['tl'][0] <= x <= rect['br'][0] and rect['tl'][1] <= y <= rect['br'][1]:
                rectangles.remove(rect)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
print("camera starting...")
cap = cv2.VideoCapture(0)
cap.set(4, 1920) 
cap.set(3, 1080) 
rectangles = {'1': [(387, 359), (534, 453)], '2': [(547, 358), (722, 452)], '3': [(362, 536), (536, 681)], '4': [(558, 536), (767, 684)]}
current_rectangle_id = 4
drawing = False
current_rectangle = [(0, 0), (0, 0)]
cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', draw_rectangle)
counter = {}
start_time = cv2.getTickCount()
frame_counter = 0

with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    for rect, val in rectangles.items():
      # cv2.rectangle(image, rect[0], rect[1], (0, 255, 0), 2)
      cv2.rectangle(image, val[0], val[1], (0, 255, 0), 2)
      cv2.putText(image, str(rect),  (val[0][0], val[0][1] - 5),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    if drawing:
      cv2.rectangle(image, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image_height, image_width, _ = image.shape
    left_hand_coord = None
    right_hand_coord = None
    if(results.pose_landmarks):
      if(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].visibility>0.8):
        left_hand_coord = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * image_width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * image_height]
      if(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].visibility>0.8):
        right_hand_coord = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * image_width, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * image_height]
    # Draw the pose annotation on the image.
    if left_hand_coord is not None or right_hand_coord is not None:
        for rect, val in rectangles.items():
            if (left_hand_coord is not None and 
                left_hand_coord[0] > val[0][0] and left_hand_coord[0] < val[1][0] and 
                left_hand_coord[1] > val[0][1] and left_hand_coord[1] < val[1][1]):
                print("LEFT HAND IS PICKING BOX NUMBER", rect)
                if(rect in counter):
                  for key in list(counter.keys()):
                    if key != rect:
                        del counter[key]
                  counter[rect] += 1
                else:
                   counter[rect] = 1
                if(counter[rect] > 4):
                  cv2.rectangle(image, val[0], val[1], (0, 0, 255), 2)
                  if(counter[rect] > 20):
                     del counter[rect]
            elif (right_hand_coord is not None and 
                  right_hand_coord[0] > val[0][0] and right_hand_coord[0] < val[1][0] and 
                  right_hand_coord[1] > val[0][1] and right_hand_coord[1] < val[1][1]):
                print("RIGHT HAND IS PICKING BOX NUMBER", rect)
                if(rect in counter):
                  for key in list(counter.keys()):
                    if key != rect:
                        del counter[key]
                  counter[rect] += 1
                else:
                   counter[rect] = 1
                if(counter[rect] > 4):
                  cv2.rectangle(image, val[0], val[1], (0, 0, 255), 2)
            else:
               counter[rect] = 0
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('Webcam', cv2.flip(image,1))
    cv2.imshow('Webcam', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    

cap.release()
