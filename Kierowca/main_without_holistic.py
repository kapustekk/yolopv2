import time

import cv2
import imutils

import utils
from database import Factors, Outcomes, KEYS
from face import FaceAnalysing


def configure_factors(frame):
    frame_height, frame_width = frame.shape[:2]
    font_scale = frame_width / 1700

    if len(Outcomes.ON_SCREEN) == 1 and Outcomes.ON_SCREEN[0] != 'face mesh' and Outcomes.ON_SCREEN[0] != 'pose mesh':
        utils.colorBackgroundText(frame, f'Setting {Outcomes.ON_SCREEN[0]} factor...', Factors.FONTS, font_scale * 2,
                                  (round(frame_width / 4), round(frame_height * 20 / 21)), 2, utils.RED, utils.YELLOW)
        if Outcomes.ON_SCREEN[0] == 'lips':
            KEYS[32] = "Factors.SET_THRESHOLD = True"
            if Factors.SET_THRESHOLD and Outcomes.ON_SCREEN[0] == 'lips':
                Factors.LIPS_RATIO_FACTOR = Outcomes.MOUTH_RATIO[0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD = False
                del KEYS[32]

        elif Outcomes.ON_SCREEN[0] == 'eyes':
            KEYS[32] = "Factors.SET_THRESHOLD = True"
            if Factors.SET_THRESHOLD and Outcomes.ON_SCREEN[0] == 'eyes':
                Factors.EYES_RATIO_FACTOR = Outcomes.EYES_RATIO[0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD = False
                del KEYS[32]

        elif Outcomes.ON_SCREEN[0] == 'pupils':
            KEYS[108] = "Factors.SET_THRESHOLD_LEFT = True"
            KEYS[114] = "Factors.SET_THRESHOLD_RIGHT = True"
            KEYS[117] = "Factors.SET_THRESHOLD_UP = True"
            KEYS[100] = "Factors.SET_THRESHOLD_DOWN = True"

            if Factors.SET_THRESHOLD_LEFT and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_LEFT_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_LEFT = False
                del KEYS[108]
            if Factors.SET_THRESHOLD_RIGHT and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_RIGHT_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_RIGHT = False
                del KEYS[114]
            if Factors.SET_THRESHOLD_UP and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_UP_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_UP = False
                del KEYS[117]
            if Factors.SET_THRESHOLD_DOWN and Outcomes.ON_SCREEN[0] == 'pupils':
                Factors.PUPIL_DOWN_THRESHOLD = Outcomes.PUPILS_DIRECTION_RATIO[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_DOWN = False
                del KEYS[100]

        elif Outcomes.ON_SCREEN[0] == 'head':
            KEYS[108] = "Factors.SET_THRESHOLD_LEFT = True"
            KEYS[114] = "Factors.SET_THRESHOLD_RIGHT = True"
            KEYS[117] = "Factors.SET_THRESHOLD_UP = True"
            KEYS[100] = "Factors.SET_THRESHOLD_DOWN = True"
            if Factors.SET_THRESHOLD_LEFT and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_LEFT_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_LEFT = False
                del KEYS[108]
            if Factors.SET_THRESHOLD_RIGHT and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_RIGHT_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][1]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_RIGHT = False
                del KEYS[114]
            if Factors.SET_THRESHOLD_UP and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_UP_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_UP = False
                del KEYS[117]
            if Factors.SET_THRESHOLD_DOWN and Outcomes.ON_SCREEN[0] == 'head':
                Factors.HEAD_DOWN_THRESHOLD = Outcomes.HEAD_POSITION_ANGLES[0][0]
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD_DOWN = False
                del KEYS[100]
        elif Outcomes.ON_SCREEN[0] == 'hands':
            KEYS[32] = "Factors.SET_THRESHOLD = True"
            if Factors.SET_THRESHOLD and Outcomes.ON_SCREEN[0] == 'hands':
                Factors.HAND_FACE_DISTANCE_FACTOR = min(Outcomes.HANDS_FACE_RATIO[0])
                Factors.CONF_MODE = False
                Factors.SET_THRESHOLD = False
                del KEYS[32]



    else:
        utils.colorBackgroundText(frame, f'Choose one body part', Factors.FONTS, font_scale * 2,
                                  (round(frame_width / 4), round(frame_height * 20 / 21)), 2, utils.RED, utils.YELLOW)


def show_fps(frame, fps):
    frame_height, frame_width = frame.shape[:2]
    print(frame_height, frame_width)
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


def main():
    # url = "http://10.160.34.153:8080/shot.jpg"
    camera = cv2.VideoCapture("uszy.mp4")
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()
    frame_counter = 0
    Outcomes.ON_SCREEN = ['pupils']
    i = 0
    face_analyzer = FaceAnalysing()


    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        # img_resp = requests.get(url)  #
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)  #
        # frame = cv2.imdecode(img_arr, -1)  ### Getting response from online camera
        # frame = imutils.resize(frame, width=1420)  #

        frame = imutils.resize(frame, width=600)  # max 1920 - min 1000
        if frame_counter % Factors.OPTIMIZATION_FACTOR == 0:  # Wpływ na optymalizację kodu -> pomijanie klatek
            i += 1
            face_analyzer.initialize_frame(frame)

            if Factors.AVERAGING_FACTOR == 1:  # ---------Brak uśredniania wyników -> chaotyczne wartości-----------
                if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                    Outcomes.IS_FACE_DETECTED = True
                    face_analyzer.get_all_ratios()
                    face_analyzer.get_all_outcomes()
                else:
                    Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

            else:
                if i % Factors.AVERAGING_FACTOR != 0:  # ---------Uśrednianie wyników-----------
                    if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                        Outcomes.IS_FACE_DETECTED = True
                        face_analyzer.get_all_ratios()

                    else:
                        Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy
                else:
                    if face_analyzer.landmarks_coords:  # Jeśli znaleziono twarz
                        Outcomes.IS_FACE_DETECTED = True
                        face_analyzer.get_all_ratios()
                        face_analyzer.get_all_outcomes()

                    else:
                        Outcomes.IS_FACE_DETECTED = False  # Jeśli nie znaleziono twarzy

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        inp = cv2.waitKey(1)
        if inp in KEYS:
            exec(KEYS[inp])
        inp = None

        if face_analyzer.landmarks_coords:
            frame = face_analyzer.draw_indicators(Outcomes.ON_SCREEN)

        if Factors.CONF_MODE:
            configure_factors(frame)

        frame = show_fps(frame, fps)

        cv2.imshow('frame', frame)

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()
