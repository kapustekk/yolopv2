import cv2


class Indices:
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
            39,
            37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    UPPER_LIP = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 282,
                348]  # ostatnie dwa landmarki to brew i polik
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    LEFT_IRIS = [474, 475, 476, 477]

    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 52,
                 119]  # ostatnie dwa landmarki to brew i polik
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_IRIS = [469, 470, 471, 472]

    # Holistic_pose
    HOL_LEFT_EYE = [1, 2, 3]
    HOL_RIGHT_EYE = [4, 5, 6]
    LEFT_SHOULDER = [11]
    RIGHT_SHOULDER = [12]
    LEFT_HAND = [15, 17, 19, 21]
    RIGHT_HAND = [16, 18, 20, 22]
    MOUTH = [9, 10]

    # Holistic_hands
    WRIST = [0]
    THUMB = [1, 2, 3, 4]
    INDEX = [5, 6, 7, 8]
    MIDDLE = [9, 10, 11, 12]
    RING = [13, 14, 15, 16]
    PINKY = [17, 18, 19, 20]


class Factors:
    EYES_RATIO_FACTOR = 0.25  # 35+ to już na maksa otwarte mniej więcej
    LIPS_RATIO_FACTOR = 0.55
    HAND_FACE_DISTANCE_FACTOR = 2.4
    # HAND_FACE_THRESHOLD_TIME = 30
    # CLOSED_EYES_FRAME = 14
    # BLINKED_EYES_FRAME = 2
    OPENED_MOUTH_FRAME = 10

    PUPIL_LEFT_THRESHOLD = -5.5
    PUPIL_RIGHT_THRESHOLD = 7.5
    PUPIL_UP_THRESHOLD = 4
    PUPIL_DOWN_THRESHOLD = 0.6

    HEAD_HORIZONTAL_THRESHOLD = 6
    HEAD_VERTICAL_THRESHOLD = 6  # Głowa ma kierunki na odwrót!!! Teraz jest dobrze

    HEAD_LEFT_THRESHOLD = -0.5
    HEAD_RIGHT_THRESHOLD = -8.5
    HEAD_UP_THRESHOLD = 7.8
    HEAD_DOWN_THRESHOLD = 4.4

    # Keys control
    CONF_MODE = False
    SET_THRESHOLD = False
    SET_THRESHOLD_UP = False
    SET_THRESHOLD_DOWN = False
    SET_THRESHOLD_LEFT = False
    SET_THRESHOLD_RIGHT = False

    AVERAGING_FACTOR = 1
    OPTIMIZATION_FACTOR = 1

    FONTS = cv2.FONT_HERSHEY_COMPLEX


class Counters:
    CLOSED_EYES_COUNTER = 0
    OPENED_MOUTH_COUNTER = 0
    HAND_FACE_FRAME_COUNTER = 0
    TOTAL_BLINKS = 0
    TOTAL_CLOSINGS = 0


class Outcomes:
    IS_FACE_DETECTED = False
    IS_POSE_DETECTED = False

    EYES_RATIO = []
    ARE_EYES_CLOSED = False

    MOUTH_RATIO = []
    IS_MOUTH_OPEN = False
    IS_YAWNING = False

    HANDS_FACE_RATIO = []
    HAND_FACE_RELATION = ''
    ARE_HANDS_CLOSE = False

    PUPILS_DIRECTION_RATIO = []
    LOOKING_DIRECTION = ''

    HEAD_POSITION_ANGLES = []
    HEAD_POSITION = ''

    SHOW_INDICATORS = False
    ON_SCREEN = []


KEYS = {
    49: "add_remove(Outcomes.ON_SCREEN,'face mesh')",  # key '1'
    50: "add_remove(Outcomes.ON_SCREEN,'eyes')",  # key '2'
    51: "add_remove(Outcomes.ON_SCREEN,'pupils')",  # key '3'
    52: "add_remove(Outcomes.ON_SCREEN,'lips')",  # key '4'
    53: "add_remove(Outcomes.ON_SCREEN,'head')",  # key '5'
    54: "add_remove(Outcomes.ON_SCREEN,'pose mesh')",  # key '6'
    55: "add_remove(Outcomes.ON_SCREEN,'hands')",  # key '7'
    56: "cv2.imwrite('wynik2.jpg', frame)",  # key '8'
    99: "Factors.CONF_MODE = not Factors.CONF_MODE", # key 'c'
    8: "Outcomes.ON_SCREEN = []", # key 'Backspace'
    # 32: "Factors.SET_THRESHOLD = True",
    # 117: "Factors.SET_THRESHOLD_UP = True",  # u UP
    # 100: "Factors.SET_THRESHOLD_DOWN = True",  # d DOWN
    # 108: "Factors.SET_THRESHOLD_LEFT = True",  # l LEFT
    # 114: "Factors.SET_THRESHOLD_RIGHT = True",  # r RIGHT
    27: "sys.exit()"  # key 'esc'
}
