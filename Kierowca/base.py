import math

import cv2
import mediapipe as mp


class MediaPipeAnalysing:
    def __init__(self, solution):
        self.solution = solution
        self.img = None
        self.frame_height = None
        self.frame_width = None
        self.results = None
        self.landmarks_coords = None
        self.mesh = self.get_mesh()

    def get_mesh(self):
        if self.solution == 'Face':
            mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                                   refine_landmarks=True, max_num_faces=1)
        elif self.solution == 'Pose':
            mesh = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            mesh = None
        return mesh

    def get_results(self):
        frame_height, frame_width = self.img.shape[:2]
        rgb_frame = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        results = self.mesh.process(rgb_frame)
        return results, frame_height, frame_width

    def get_landmarks_coords(self):
        if self.solution == "Face":
            coords = [(int(point.x * self.frame_width), int(point.y * self.frame_height)) for point in
                      self.results.multi_face_landmarks[0].landmark]
        elif self.solution == 'Pose':
            coords = [(int(point.x * self.frame_width), int(point.y * self.frame_height)) for point in
                      self.results.pose_landmarks.landmark]
        else:
            coords = None
        return coords

    def initialize_frame(self, img):
        self.img = img
        self.results, self.frame_height, self.frame_width = self.get_results()
        try:
            self.landmarks_coords = self.get_landmarks_coords()
        except:
            self.landmarks_coords = None

    def get_average_point(self, indices):
        x, y = 0, 0
        for point in indices:
            x += self.landmarks_coords[point][0]
            y += self.landmarks_coords[point][1]

        average_coords = [round(x / len(indices)), round(y / len(indices))]
        return average_coords


    @staticmethod
    def get_euclidean_distance(point1, point2):
        x, y = point1
        x1, y1 = point2
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance
