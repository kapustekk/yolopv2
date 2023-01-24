import time
import numpy as np
import cv2
import imutils

from Kierowca import utils
from Kierowca.database import Factors, Outcomes, KEYS
from Kierowca.face import FaceAnalysing
from Kierowca.pose import PoseAnalysing
import requests

def interpret_data():
    pass


def show_fps(frame, fps):
    frame_height, frame_width = frame.shape[:2]
    font_scale = frame_width / 1700
    frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', Factors.FONTS, font_scale,
                                     (round(frame_width / 100), round(frame_height / 20)), bgOpacity=0.9,
                                     textThickness=2)

    utils.colorBackgroundText(frame, f'Face : {Outcomes.IS_FACE_DETECTED}', Factors.FONTS, font_scale * 0.8,
                              (round(frame_width / 100), round(frame_height / 10)), 2, utils.BLACK, utils.WHITE)
    utils.colorBackgroundText(frame, f'Pose : {Outcomes.IS_POSE_DETECTED}', Factors.FONTS, font_scale * 0.8,
                              (round(frame_width / 100), round(frame_height * 2 / 15)), 2, utils.BLACK, utils.WHITE)

    return frame


def add_remove(li, elem):
    if elem not in li:
        li.append(elem)
    else:
        li.remove(elem)
    return li


def main(ret,frame,pose_analyzer,face_analyzer,inp):
    #url = "http://10.84.142.170:8080/shot.jpg"
    start_time = time.time()
    frame_counter = 1
    i = 0
    #frame = cv2.flip(frame,1)

    #pose_analyzer = PoseAnalysing()
    # output = cv2.VideoWriter(“path”, cv2.VideoWriter_fourcc(*’MPEG’), 30, (1080, 1920))



    if ret:
        # img_resp = requests.get(url)
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # frame = cv2.imdecode(img_arr, -1)
        # frame = imutils.resize(frame, width=1920)

        frame = imutils.resize(frame, width=720)  # max 1920 - min 1000
        if frame_counter % Factors.OPTIMIZATION_FACTOR == 0:  # Wpływ na optymalizację kodu -> pomijanie klatek
            i += 1
            face_analyzer.initialize_frame(frame)
            #pose_analyzer.initialize_frame(frame)

            if Factors.AVERAGING_FACTOR == 1:  # ---------Brak uśredniania wyników -> chaotyczne wartości-----------
                if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                    Outcomes.IS_FACE_DETECTED = True
                    face_analyzer.get_all_ratios()
                    face_analyzer.get_all_outcomes()

                else:
                    Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

                # if pose_analyzer.landmarks_coords:  # Jeśli znaleziono ciało
                #     Outcomes.IS_POSE_DETECTED = True
                #     pose_analyzer.get_all_ratios()
                #     pose_analyzer.get_all_outcomes()
                # else:
                #     Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała

            else:
                if i % Factors.AVERAGING_FACTOR != 0:  # ---------Uśrednianie wyników-----------

                    if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                        Outcomes.IS_FACE_DETECTED = True
                        face_analyzer.get_all_ratios()

                    else:
                        Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

                    # if pose_analyzer.landmarks_coords:  # Jeśli znaleziono ciało
                    #     Outcomes.IS_POSE_DETECTED = True
                    #     pose_analyzer.get_all_ratios()
                    #
                    # else:
                    #     Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała
                else:
                    if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                        Outcomes.IS_FACE_DETECTED = True
                        face_analyzer.get_all_ratios()
                        face_analyzer.get_all_outcomes()

                    else:
                        Outcomes.IS_FACE_DETECTED = False  # # Jeśli nie znaleziono twarzy

                    # if pose_analyzer.landmarks_coords:  # Jeśli znaleziono ciało
                    #     Outcomes.IS_POSE_DETECTED = True
                    #     pose_analyzer.get_all_ratios()
                    #     pose_analyzer.get_all_outcomes()

                    # else:
                    #     Outcomes.IS_POSE_DETECTED = False  # Jeśli nie znaleziono ciała

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        # frame_height, frame_width = frame.shape[:2]
        # Outcomes.OBJECTS = ['eyes']
        if inp in KEYS:
            exec(KEYS[inp])
        inp = None

        if face_analyzer.landmarks_coords:
            frame = face_analyzer.draw_indicators(Outcomes.OBJECTS)
        # if pose_analyzer.landmarks_coords:
        #     frame = pose_analyzer.draw_indicators(Outcomes.OBJECTS6, frame)

        frame = show_fps(frame, fps)

        cv2.imshow('frame', frame)
        #cv2.imwrite('wynik1.jpg', frame)

    #cv2.destroyAllWindows()
    #camera.release()


if __name__ == '__main__':
    main()
