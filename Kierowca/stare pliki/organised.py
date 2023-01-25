import math
import time

import cv2
import mediapipe as mp
import numpy as np

from database import Indices, Factors, Counters, Outcomes
import utils3 as utils

# variables
frame_counter = 0
CLOSED_EYES_COUNTER = 0
OPENED_MOUTH_COUNTER = 0
TOTAL_BLINKS = 0
TOTAL_CLOSINGS = 0

# constants
EYES_RATIO_FACTOR = 0.25  # 35+ to już na maksa otwarte mniej więcej
LIPS_RATIO_FACTOR = 0.3
HAND_FACE_DISTANCE_FACTOR = 150
HAND_FACE_THRESHOLD_TIME = 30
CLOSED_EYES_FRAME = 14
BLINKED_EYES_FRAME = 2
OPENED_MOUTH_FRAME = 10
ARE_EYES_CLOSED = False
IS_SLEEPING = False
IS_YAWNING = False
IS_MOUTH_OPEN = False
IS_FACE_DETECTED = False
ARE_HANDS_DETECTED = False
ARE_HANDS_CLOSE = False
FONTS = cv2.FONT_HERSHEY_COMPLEX

# camera object
camera = cv2.VideoCapture("jebs.mp4")


# landmark detection function
def landmarksDetection(img, res, type, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    # print(res.multi_face_landmarks[0].landmark)
    if type == "Face":
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      res.multi_face_landmarks[0].landmark]
    else:
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      res.pose_landmarks.landmark]
    if draw:
        [cv2.circle(img, p, 2, (255, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclaidean distance
def euclaidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


def get_average_point(img, landmarks, indices):
    x, y = 0, 0
    for point in indices:
        x += landmarks[point][0]
        y += landmarks[point][1]
        # cv2.circle(img, (landmarks[point][0], landmarks[point][1]), 2, (255, 255, 255), -1)

    average_coords = [round(x / len(indices)), round(y / len(indices))]
    return average_coords


def estimate_hand_face_distance(img, landmarks, left_hand_indices, right_hand_indices):
    hand_face_frame_counter = 0
    left_hand_point = get_average_point(img, landmarks, left_hand_indices)
    right_hand_point = get_average_point(img, landmarks, right_hand_indices)

    cv2.circle(img, right_hand_point, 10, (255, 255, 255), -1)
    cv2.circle(img, left_hand_point, 10, (0, 0, 255), -1)

    left_ear_point = landmarks[7]
    right_ear_point = landmarks[8]
    # mouth_point = landmarks[9]
    # face_point = landmarks[0]

    ll_distance = euclaidean_distance(left_hand_point, left_ear_point)
    rr_distance = euclaidean_distance(right_hand_point, right_ear_point)
    lr_distance = euclaidean_distance(left_hand_point, right_ear_point)
    rl_distance = euclaidean_distance(right_hand_point, left_ear_point)

    for distance in [ll_distance, rr_distance, lr_distance, rl_distance]:
        if distance < HAND_FACE_DISTANCE_FACTOR:
            hand_face_frame_counter += 1
            if frame_counter > HAND_FACE_THRESHOLD_TIME:
                is_hand_close = True
        else:
            is_hand_close = False
            hand_face_frame_counter = 0

    print(distance)
    return is_hand_close


# Mouth Ratio
def estimate_lips_ratio(landmarks, upper_indices, lower_indices):
    upper_lip = landmarks[upper_indices[13]]
    lower_lip = landmarks[lower_indices[16]]
    left_corner = landmarks[lower_indices[0]]
    right_cornet = landmarks[lower_indices[10]]

    vdistance = euclaidean_distance(upper_lip, lower_lip)
    hdistance = euclaidean_distance(left_corner, right_cornet)

    ratio = vdistance / hdistance
    return ratio


# Blinking Ratio
def estimate_eyes_ratio(landmarks, right_indices, left_indices):
    # Right eye
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhdistance = euclaidean_distance(rh_right, rh_left)
    rvdistance = euclaidean_distance(rv_top, rv_bottom)

    lvdistance = euclaidean_distance(lv_top, lv_bottom)
    lhdistance = euclaidean_distance(lh_right, lh_left)

    r_ratio = rvdistance / rhdistance
    l_ratio = lvdistance / lhdistance

    ratio = (r_ratio + l_ratio) / 2
    return ratio


HORIZONTAL_THRESHOLD = 2
VERTICAL_THRESHOLD = 4


def estimate_eye_position(img, landmarks, eye_iris_indices, eye_mesh_indices):
    eye_pupil_center = get_average_point(img, landmarks, eye_iris_indices)
    eye_center = get_average_point(img, landmarks, eye_mesh_indices[:-2])
    cv2.circle(img, eye_pupil_center, 2, (0, 0, 255), -1)
    cv2.circle(img, eye_center, 2, (0, 255, 0), -1)
    cv2.line(img, eye_center, eye_pupil_center, (0, 0, 255), 3)

    eye_left_border = landmarks[eye_mesh_indices[0]]
    eye_right_border = landmarks[eye_mesh_indices[8]]
    eye_top_border = landmarks[eye_mesh_indices[-1]]
    eye_bottom_border = landmarks[eye_mesh_indices[-2]]

    horizontal_eye_distance = abs(eye_left_border[0] + eye_right_border[0])
    vertical_eye_distance = abs(eye_top_border[1] + eye_bottom_border[1])

    center_left_distance = round(eye_left_border[0] - eye_pupil_center[0])
    center_right_distance = round(eye_right_border[0] - eye_pupil_center[0])
    center_top_distance = round(eye_top_border[1] - eye_pupil_center[1])
    center_bottom_distance = round(eye_bottom_border[1] - eye_pupil_center[1])

    horizontal_pupil_distance = center_left_distance + center_right_distance
    vertical_pupil_distance = center_top_distance + center_bottom_distance

    horizontal_ratio = 100 * horizontal_pupil_distance / horizontal_eye_distance
    vertical_ratio = 100 * vertical_pupil_distance / vertical_eye_distance

    pupil_direction_ratio = [horizontal_ratio, vertical_ratio]
    return pupil_direction_ratio


def get_looking_direction(left_ratios, right_ratios):
    [left_horizontal_ratio, left_vertical_ratio] = left_ratios
    [right_horizontal_ratio, right_vertical_ratio] = right_ratios
    horizontal_ratio = (left_horizontal_ratio + right_horizontal_ratio) / 2
    vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2

    looking_horizontal_direction = "Forward"
    looking_vertical_direction = "Forward"
    if horizontal_ratio < -HORIZONTAL_THRESHOLD:
        looking_horizontal_direction = "Left"
    elif horizontal_ratio > HORIZONTAL_THRESHOLD:
        looking_horizontal_direction = "Right"

    if vertical_ratio > VERTICAL_THRESHOLD:
        looking_vertical_direction = "Up"
    elif vertical_ratio < -VERTICAL_THRESHOLD:
        looking_vertical_direction = "Down"

    eye_looking_direction = [looking_horizontal_direction, looking_vertical_direction]
    return eye_looking_direction


def detect_yawning(ratio, opened_mouth_counter, is_yawning):
    if ratio > LIPS_RATIO_FACTOR:
        opened_mouth_counter += 1
        # utils.colorBackgroundText(frame, f'Opened mouth for: {opened_mouth_counter}', FONTS, 1.7,
        #                           (int(frame_height / 2), 200), 2, utils.YELLOW, pad_x=6, pad_y=6)

        if opened_mouth_counter > OPENED_MOUTH_FRAME:
            is_yawning = True
    else:
        is_yawning = False
        opened_mouth_counter = 0

    return opened_mouth_counter, is_yawning


def detect_sleeping(eyes_ratio, closed_eyes_counter, is_sleeping, is_closed, total_blinks, total_closings):
    if eyes_ratio < EYES_RATIO_FACTOR:  # komentarz, okej?!
        closed_eyes_counter += 1
        # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
        # utils.colorBackgroundText(frame, f'Closed eyes for: {closed_eyes_counter}', FONTS, 1.7,
        #                           (int(frame_height / 2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6)
        if closed_eyes_counter > CLOSED_EYES_FRAME:
            is_closed = True
        if closed_eyes_counter > 3 * CLOSED_EYES_FRAME:
            is_sleeping = True
            utils.colorBackgroundText(frame, f'SLEEP ALERT', FONTS, 1.7,
                                      (int(frame_height / 2), 500), 3, utils.RED, pad_x=6, pad_y=6)
    else:
        if is_sleeping:
            is_sleeping = False
            closed_eyes_counter = 0

        if is_closed:
            is_closed = False
            total_closings += 1

        elif BLINKED_EYES_FRAME < closed_eyes_counter < CLOSED_EYES_FRAME:
            total_blinks += 1
            closed_eyes_counter = 0

    return closed_eyes_counter, is_sleeping, is_closed, total_blinks, total_closings


def estimate_head_orientation(mesh_landmarks, draw=False):
    cv2FONTS = cv2.FONT_HERSHEY_COMPLEX
    dir_treshold = 6
    face_3d = []
    face_2d = []
    position = []

    for face_landmarks in mesh_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * frame_width, lm.y * frame_height)

                x, y = int(lm.x * frame_width), int(lm.y * frame_height)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * frame_width

        cam_matrix = np.array([[focal_length, 0, frame_height / 2],
                               [0, focal_length, frame_width / 2],
                               [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        position.append(angles[0] * 360)
        position.append(angles[1] * 360)
        position.append(angles[2] * 360)

        # See where the user's head tilting
        if position[1] < -dir_treshold:
            direction = "Facing Left"
        elif position[1] > dir_treshold:
            direction = "Facing Right"
        elif position[0] < -dir_treshold:
            direction = "Facing Down"
        elif position[0] > dir_treshold:
            direction = "Facing Up"
        else:
            direction = "Forward"
        position.append(direction)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + position[1] * 10), int(nose_2d[1] - position[0] * 10))
        if draw:
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            cv2.putText(frame, direction, (20, 50), cv2FONTS, 2, (0, 255, 0), 2)
            cv2.putText(frame, "x: " + str(np.round(position[0], 2)), (500, 50), cv2FONTS, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(np.round(position[1], 2)), (500, 100), cv2FONTS, 1, (0, 0, 255), 2)
            cv2.putText(frame, "z: " + str(np.round(position[2], 2)), (500, 150), cv2FONTS, 1, (0, 0, 255), 2)
    return position


def show_indicators(show_eyes=False, show_blinks=False, show_yawns=False):
    if show_eyes:
        cv2.putText(frame, f'Eyes ratio {round(eyes_ratio, 2)}', (30, 100), FONTS, 1.0, utils.GREEN, 2)
        # utils.colorBackgroundText(frame, f'Ratio : {round(eyes_ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
        # utils.YELLOW)
        utils.colorBackgroundText(frame, f'Looking {looking_direction}', FONTS, 1.0, (40, 450), 2, utils.BLACK, utils.WHITE,
                                  8, 8)
    if show_blinks:
        # cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONTS, 0.6, utils.GREEN, 2)
        # cv2.putText(frame, f'Total Closings: {TOTAL_CLOSINGS}', (30, 200), FONTS, 0.6, utils.GREEN, 2)
        utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 200), 2)
        utils.colorBackgroundText(frame, f'Total Closings: {TOTAL_CLOSINGS}', FONTS, 0.7, (30, 250), 2)

    if show_yawns:
        utils.colorBackgroundText(frame, f'Mouth ratio: {round(lips_ratio, 2)}', FONTS, 0.7, (30, 150), 2)

    # if show_head_orientation:
    #     pass


if __name__ == '__main__':
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                                refine_landmarks=True)
    start_time = time.time()

    while True:
        frame_counter += 1  # frame counter
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        hol_results = holistic.process(rgb_frame)

        if results.multi_face_landmarks:
            head_position = estimate_head_orientation(results.multi_face_landmarks, True)
            mesh_coords = landmarksDetection(frame, results, "Face", False)

            lips_ratio = estimate_lips_ratio(mesh_coords, Indices.UPPER_LIP, Indices.LOWER_LIP)
            OPENED_MOUTH_COUNTER, IS_YAWNING = detect_yawning(lips_ratio, OPENED_MOUTH_COUNTER, IS_YAWNING)

            eyes_ratio = estimate_eyes_ratio(mesh_coords, Indices.RIGHT_EYE, Indices.LEFT_EYE)
            CLOSED_EYES_COUNTER, IS_SLEEPING, ARE_EYES_CLOSED, TOTAL_BLINKS, TOTAL_CLOSINGS = detect_sleeping(eyes_ratio,
                                                                                                              CLOSED_EYES_COUNTER,
                                                                                                              IS_SLEEPING,
                                                                                                              ARE_EYES_CLOSED,
                                                                                                              TOTAL_BLINKS,
                                                                                                              TOTAL_CLOSINGS)

            if not ARE_EYES_CLOSED:
                right_eye_position = estimate_eye_position(frame, mesh_coords, Indices.LEFT_IRIS,
                                                           Indices.LEFT_EYE)
                left_eye_position = estimate_eye_position(frame, mesh_coords, Indices.RIGHT_IRIS,
                                                          Indices.RIGHT_EYE)
                looking_direction = get_looking_direction(left_eye_position, right_eye_position)

            show_indicators(show_eyes=True, show_blinks=True, show_yawns=True)

        else:
            utils.textBlurBackground(frame, 'No face detected!', FONTS, 0.9,
                                     (int(frame_width / 2), int(frame_height / 2)),
                                     2, (0, 0, 255), (49, 49), 13, 13)
        if hol_results.pose_landmarks:
            holistic_coords = landmarksDetection(frame, hol_results, "Pose", False)
            ARE_HANDS_CLOSE = estimate_hand_face_distance(frame, holistic_coords, Indices.LEFT_HAND,
                                                          Indices.RIGHT_HAND)

        else:
            pass
            # utils.textBlurBackground(frame, 'No hands detected!', FONTS, 0.9,
            #                          (int(frame_width / 2), int(frame_height / 2) + 50),
            #                          2, (0, 0, 255), (49, 49), 13, 13)





        # cv2.polylines(frame, [mesh_coords[faceindices.LEFT_IRIS]], True, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.polylines(frame, [mesh_coords[faceindices.RIGHT_IRIS]], True, (0, 255, 0), 1, cv2.LINE_AA)

        # for p in faceindices.LEFT_IRIS:
        #     frame = cv2.circle(frame, mesh_coords[p], 2, (255, 255, 0), -1)

        # estimate_eye_position2(frame, mesh_coords, faceindices.LEFT_IRIS, faceindices.LEFT_EYE)
        # # # Right hand
        # mp.solutions.drawing_utils.draw_landmarks(frame, hol_results.right_hand_landmarks,
        #                                           mp.solutions.holistic.HAND_CONNECTIONS)
        #
        # # Left Hand
        # mp.solutions.drawing_utils.draw_landmarks(frame, hol_results.left_hand_landmarks,
        #                                           mp.solutions.holistic.HAND_CONNECTIONS)
        #
        # # Pose Detections
        # mp.solutions.drawing_utils.draw_landmarks(frame, hol_results.pose_landmarks,
        #                                           mp.solutions.holistic.POSE_CONNECTIONS)

        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        end_time = time.time() - start_time
        fps = frame_counter / end_time
        cv2.imshow('frame', frame)
        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                         textThickness=2)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    camera.release()
