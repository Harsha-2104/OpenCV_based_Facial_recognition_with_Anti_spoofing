import cv2
import numpy as np
import face_recognition
import os
import xlwt, xlrd
from xlwt import Workbook
from xlutils.copy import copy as xl_copy
from datetime import date
import time
import dlib
from scipy.spatial import distance

# Flags & Counters for Liveness Checks
consecutive_successful_checks = 0
required_successful_checks = 1
real_face_hits = 0

# Load facial landmarks predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# Eye Aspect Ratio Calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESHOLD = 0.215
LIP_DISTANCE_THRESHOLD = 10
HEAD_MOVEMENT_THRESHOLD = 10
last_nose_position = None

# Liveness check functions
def detect_eye_blink(landmarks):
    left_eye = np.array([(landmarks.part(i).x * 4, landmarks.part(i).y * 4) for i in range(36, 42)])
    right_eye = np.array([(landmarks.part(i).x * 4, landmarks.part(i).y * 4) for i in range(42, 48)])

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0

    print(f"EAR: {ear:.3f}")
    
    if ear < EYE_AR_THRESHOLD:
        return True
    return False

def detect_lip_movement(landmarks):
    top_lip = np.array([(landmarks.part(i).x * 4, landmarks.part(i).y * 4) for i in range(50, 53)] + [(landmarks.part(i).x * 4, landmarks.part(i).y * 4) for i in range(61, 64)])
    bottom_lip = np.array([(landmarks.part(i).x * 4, landmarks.part(i).y * 4) for i in range(56, 59)] + [(landmarks.part(i).x * 4, landmarks.part(i).y * 4) for i in range(65, 68)])

    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)

    distance_between_lips = abs(top_mean[1] - bottom_mean[1])

    return distance_between_lips > LIP_DISTANCE_THRESHOLD

def detect_head_movement(landmarks):
    global last_nose_position

    nose_point = (landmarks.part(30).x * 4, landmarks.part(30).y * 4)

    if last_nose_position is None:
        last_nose_position = nose_point
        return False

    distance_moved = distance.euclidean(nose_point, last_nose_position)
    last_nose_position = nose_point

    return distance_moved > HEAD_MOVEMENT_THRESHOLD

def run_liveness_checks(landmarks):
    return detect_eye_blink(landmarks) and detect_lip_movement(landmarks) and detect_head_movement(landmarks)

# Load known faces
CurrentFolder = os.getcwd()
images_folder = os.path.join(CurrentFolder, 'faces')
known_face_encodings = []
known_face_names = []

for file in os.listdir(images_folder):
    if file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(images_folder, file)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(file)[0])

# Initialize Excel file
rb = xlrd.open_workbook('attendence_excel.xls', formatting_info=True)
wb = xl_copy(rb)
inp = input('Please give file name: ')
sheet1 = wb.add_sheet(str(date.today()) + ',' + inp)
sheet1.write(0, 0, 'ID NO.')
sheet1.write(0, 1, 'Attended Time')
row = 1
already_attendence_taken = []

# Video capture setup
video_capture = cv2.VideoCapture(0)
FRAME_PROCESSING_FREQUENCY = 5
DELAY_BETWEEN_FRAMES = 0.1  
frame_counter = 0

while True:
    ret, frame = video_capture.read()
    time.sleep(DELAY_BETWEEN_FRAMES)  
    frame_counter += 1

    if frame_counter % FRAME_PROCESSING_FREQUENCY != 0:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    for (top, right, bottom, left) in face_locations:
        face_rect = dlib.rectangle(left, top, right, bottom)
        landmarks = predictor(gray, face_rect)

        if run_liveness_checks(landmarks):
            face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_locations[0]])[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[best_match_index]

            if name not in already_attendence_taken and name != "Unknown":
                sheet1.write(row, 0, name)
                sheet1.write(row, 1, time.ctime())
                row += 1
                already_attendence_taken.append(name)
                wb.save('attendence_excel.xls')


            cv2.putText(frame, f"Real Face Detected: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Real Face Detected: {name}")
        else:
            cv2.putText(frame, "Fake Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

wb.save('attendence_excel.xls')
video_capture.release()
cv2.destroyAllWindows()
