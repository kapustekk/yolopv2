import cv2
import mediapipe as mp

import utils
from Kierowca.base import MediaPipeAnalysing
from Kierowca.database import Indices, Factors, Outcomes


class PoseAnalysing(MediaPipeAnalysing):
    def __init__(self):
        super().__init__('Pose')
        self.solution = 'Pose'
        self.hands_face_ratio = None

    def estimate_hand_face_distance_ratios(self, draw=False):
        left_hand_indices = Indices.LEFT_HAND
        right_hand_indices = Indices.RIGHT_HAND

        coords = [(int(point.x * self.frame_width), int(point.y * self.frame_height), int(point.z * self.frame_width))
                  for point in
                  self.results.pose_landmarks.landmark]


        left_hand_point = self.get_average_3d_point(left_hand_indices, coords)
        right_hand_point = self.get_average_3d_point(right_hand_indices, coords)

        left_ear_point = coords[7]  # Left ear
        right_ear_point = coords[8]  # Right ear

        ear2ear = self.get_euclidean_3d_distance(left_ear_point, right_ear_point)

        ll_distance = self.get_euclidean_3d_distance(left_hand_point, left_ear_point)
        rr_distance = self.get_euclidean_3d_distance(right_hand_point, right_ear_point)
        lr_distance = self.get_euclidean_3d_distance(left_hand_point, right_ear_point)
        rl_distance = self.get_euclidean_3d_distance(right_hand_point, left_ear_point)

        hand_face_distance_ratios = [ll_distance / ear2ear, rr_distance / ear2ear, lr_distance / ear2ear,
                                     rl_distance / ear2ear]
        if draw:
            cv2.circle(self.img, left_hand_point[:2], 4, utils.YELLOW, 8)
            cv2.circle(self.img, right_hand_point[:2], 4, utils.ORANGE, -1)
            cv2.circle(self.img, left_ear_point[:2], 4, utils.MAGENTA, -1)
            cv2.circle(self.img, right_ear_point[:2], 4, utils.PINK, -1)

            cv2.line(self.img, left_hand_point[:2], left_ear_point[:2], utils.YELLOW, 2)
            cv2.line(self.img, right_hand_point[:2], right_ear_point[:2], utils.ORANGE, 2)
            cv2.line(self.img, left_hand_point[:2], right_ear_point[:2], utils.MAGENTA, 2)
            cv2.line(self.img, right_hand_point[:2], left_ear_point[:2], utils.PINK, 2)

        return hand_face_distance_ratios

    def get_hand_face_position(self):
        ratios = self.hands_face_ratio
        Outcomes.ARE_HANDS_CLOSE = True
        if min(ratios) < Factors.HAND_FACE_DISTANCE_FACTOR:
            hand_face = "Hand is too close to the ear"
        # if ratios[0] < Factors.HAND_FACE_DISTANCE_FACTOR:
        #     hand_face = 'Left hand is too close to left ear'
        # elif ratios[1] < Factors.HAND_FACE_DISTANCE_FACTOR:
        #     hand_face = 'Right hand is too close to right ear'
        # elif ratios[2] < Factors.HAND_FACE_DISTANCE_FACTOR:
        #     hand_face = 'Left hand is too close to right ear'
        # elif ratios[3] < Factors.HAND_FACE_DISTANCE_FACTOR:
        #     hand_face = 'Right hand is too close to left ear'
        else:
            hand_face = "It's okay"
            Outcomes.ARE_HANDS_CLOSE = False

        return hand_face

    def get_all_ratios(self):
        Outcomes.HANDS_FACE_RATIO.append(self.estimate_hand_face_distance_ratios())

    def get_all_outcomes(self):
        hand_ratios = Outcomes.HANDS_FACE_RATIO
        self.hands_face_ratio = list(map(lambda *x: sum(x) / len(hand_ratios), *hand_ratios))
        Outcomes.HAND_FACE_RELATION = self.get_hand_face_position()
        self.hands_face_ratio = [round(x, 1) for x in self.hands_face_ratio]
        Outcomes.HANDS_FACE_RATIO = []

    # def estimate_head_orientation2(self): TODO nie wiem czy to ma sens, zbyt proste, może nie działać w wielu sytuacjach
    #     left_eye_point = self.landmarks_coords[Indices.HOL_LEFT_EYE[1]]
    #     right_eye_point = self.landmarks_coords[Indices.HOL_RIGHT_EYE[1]]
    #     left_shoulder_point = self.landmarks_coords[Indices.LEFT_SHOULDER[0]]
    #     right_shoulder_point = self.landmarks_coords[Indices.RIGHT_SHOULDER[0]]
    #
    #     lr_distance = self.get_euclidean_distance(left_eye_point, right_shoulder_point)
    #     rl_distance = self.get_euclidean_distance(right_eye_point, left_shoulder_point)
    #     self.img = cv2.circle(self.img, left_eye_point, 2, (255, 255, 255), -1)
    #     shoulder_distance = self.get_euclidean_distance(left_shoulder_point, right_shoulder_point)
    #
    #
    #     print(shoulder_distance)

    def draw_indicators(self, objects, frame):
        height = 150
        f_width = round(100 / 2100 * self.frame_width)
        f_height = round(1000 / 1181 * self.frame_height)
        font_scale = self.frame_width / 2100 * 1.5

        if 'pose mesh' in objects:
            mp_drawing = mp.solutions.drawing_utils
            holistic = mp.solutions.holistic
            draw_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=1,
                                               color=(150, 0, 0))  # color=(100, 100, 100))

            if self.results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, self.results.pose_landmarks, holistic.POSE_CONNECTIONS, draw_spec,
                                          draw_spec)

        if 'hands' in objects:
            height -= 150
            self.estimate_hand_face_distance_ratios(True)

            if Outcomes.HAND_FACE_RELATION == "Hand is too close to the ear":
                color = utils.RED
                pos = 'TooClose|'
            else:
                pos = ''
                color = utils.WHITE

            utils.colorBackgroundText(frame,
                                      f'Hand-Face dist.: {pos}{self.hands_face_ratio[0]}|'
                                      f'{self.hands_face_ratio[1]}|{self.hands_face_ratio[2]}|{self.hands_face_ratio[3]}',
                                      Factors.FONTS, font_scale, (f_width, f_height + height), 2, utils.BLACK, color)
            utils.colorBackgroundText(frame, f'Threshold: {Factors.HAND_FACE_DISTANCE_FACTOR}', Factors.FONTS,
                                      font_scale,
                                      (f_width, f_height + 50 + height), 2, utils.BLACK, utils.WHITE)

        return frame
