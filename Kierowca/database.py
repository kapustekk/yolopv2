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
    LIPS_RATIO_FACTOR = 0.3
    HAND_FACE_DISTANCE_FACTOR = 4.5
    HAND_FACE_THRESHOLD_TIME = 30
    CLOSED_EYES_FRAME = 14
    BLINKED_EYES_FRAME = 2
    OPENED_MOUTH_FRAME = 10
    HORIZONTAL_THRESHOLD = 1
    VERTICAL_THRESHOLD = 2
    HEAD_DIRECTION_THRESHOLD = 6
    AVERAGING_FACTOR = 1

    FONTS = cv2.FONT_HERSHEY_COMPLEX

    OPTIMIZATION_FACTOR = 1


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

    HANDS_FACE_RATIO = []
    HAND_FACE_RELATION = False
    ARE_HANDS_CLOSE = False

    PUPILS_DIRECTION_RATIO = []
    LOOKING_DIRECTION = ''#Lelft Right Up Down

    HEAD_POSITION_ANGLES = []
    HEAD_POSITION = ''#Left Right Up Down

    SHOW_INDICATORS = False
    OBJECTS = []


KEYS = {
    49: "add_remove(Outcomes.OBJECTS,'face mesh')",     # key '1'
    50: "add_remove(Outcomes.OBJECTS,'eyes')",          # key '2'
    51: "add_remove(Outcomes.OBJECTS,'pupils')",        # key '3'
    52: "add_remove(Outcomes.OBJECTS,'lips')",          # key '4'
    53: "add_remove(Outcomes.OBJECTS,'head')",          # key '5'
    54: "add_remove(Outcomes.OBJECTS,'pose mesh')",     # key '6'
    55: "add_remove(Outcomes.OBJECTS,'hands')",         # key '7'
    56: "cv2.imwrite('wynik2.jpg', frame)",             # key '8'
    27: "sys.exit()"                                    # key 'esc'
}
