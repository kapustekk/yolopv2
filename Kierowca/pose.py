import cv2
import mediapipe as mp

from Kierowca import utils
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

        coords = [(int(point.x * self.frame_width), int(point.y * self.frame_height))
                  for point in
                  self.results.pose_landmarks.landmark]

        left_hand_point = self.get_average_point(left_hand_indices)
        right_hand_point = self.get_average_point(right_hand_indices)

        left_ear_point = coords[7]  # Left ear
        right_ear_point = coords[8]  # Right ear
        mouth_point = self.get_average_point([Indices.MOUTH[0], Indices.MOUTH[1]])
        nose_point = self.get_average_point([0])

        left_ear2nose = self.get_euclidean_distance(left_ear_point, nose_point)
        right_ear2nose = self.get_euclidean_distance(right_ear_point, nose_point)
        ear2nose = (left_ear2nose + right_ear2nose) / 2

        ll_distance = self.get_euclidean_distance(left_hand_point, left_ear_point)
        rr_distance = self.get_euclidean_distance(right_hand_point, right_ear_point)
        lr_distance = self.get_euclidean_distance(left_hand_point, right_ear_point)
        rl_distance = self.get_euclidean_distance(right_hand_point, left_ear_point)

        lm_distance = self.get_euclidean_distance(left_hand_point, mouth_point)
        rm_distance = self.get_euclidean_distance(right_hand_point, mouth_point)

        hand_face_distance_ratios = [ll_distance / ear2nose, lr_distance / ear2nose, rr_distance / ear2nose,
                                     rl_distance / ear2nose]  # lm_distance / ear2nose, rm_distance / ear2nose]

        if draw:
            cv2.circle(self.img, left_hand_point[:2], 4, utils.YELLOW, 8)
            cv2.circle(self.img, right_hand_point[:2], 4, utils.ORANGE, -1)
            cv2.circle(self.img, left_ear_point[:2], 4, utils.MAGENTA, -1)
            cv2.circle(self.img, right_ear_point[:2], 4, utils.PINK, -1)
            cv2.circle(self.img, mouth_point[:2], 4, utils.RED, -1)
            cv2.circle(self.img, nose_point[:2], 4, utils.RED, -1)

            cv2.line(self.img, left_hand_point[:2], left_ear_point[:2], utils.YELLOW, 2)
            cv2.line(self.img, right_hand_point[:2], right_ear_point[:2], utils.ORANGE, 2)
            cv2.line(self.img, left_hand_point[:2], right_ear_point[:2], utils.MAGENTA, 2)
            cv2.line(self.img, right_hand_point[:2], left_ear_point[:2], utils.PINK, 2)
            cv2.line(self.img, right_hand_point[:2], mouth_point[:2], utils.GRAY, 2)
            cv2.line(self.img, left_hand_point[:2], mouth_point[:2], utils.GRAY, 2)

        return hand_face_distance_ratios

    def get_hand_face_position(self):
        ratios = self.hands_face_ratio
        if min(ratios[:4]) < Factors.HAND_FACE_DISTANCE_FACTOR:
            hand_face = "Hand is too close to the ear"
            Outcomes.ARE_HANDS_CLOSE = True
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
        Outcomes.HANDS_FACE_RATIO.append(self.hands_face_ratio)

    def draw_indicators(self, objects, frame):
        height = 150
        f_width = round(100 / 2100 * self.frame_width)
        f_height = round(1000 / 1181 * self.frame_height)
        font_scale = self.frame_width / 2100 * 1.5

        if 'pose mesh' in objects:
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            draw_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=1,
                                               color=(150, 0, 0))  # color=(100, 100, 100))

            if self.results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec,
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

            utils.colorBackgroundText(frame, f'Hand-Face dist.: {pos}{self.hands_face_ratio[0]}|'
                                             f'{self.hands_face_ratio[1]}|{self.hands_face_ratio[2]}|'
                                             f'{self.hands_face_ratio[3]}',
                                      Factors.FONTS, font_scale, (f_width, f_height + height), 2, utils.BLACK, color)

            utils.colorBackgroundText(frame, f'Threshold: {Factors.HAND_FACE_DISTANCE_FACTOR}', Factors.FONTS,
                                      font_scale, (f_width, f_height + 50 + height), 2, utils.BLACK, utils.WHITE)

        return frame
