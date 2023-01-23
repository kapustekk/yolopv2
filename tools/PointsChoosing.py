import cv2
import numpy as np
import os.path
from numpy import genfromtxt

CHOSEN_POINTS_COUNTER = 0
CHOSEN_POINTS = np.zeros((7,2), np.int)


def find_optic_middle(CHOSEN_POINTS):
    xb = int((CHOSEN_POINTS[0][0]+CHOSEN_POINTS[1][0])/2)
    yb = int((CHOSEN_POINTS[0][1]+CHOSEN_POINTS[1][1])/2)
    xu = int((CHOSEN_POINTS[2][0]+CHOSEN_POINTS[3][0])/2)
    yu = int((CHOSEN_POINTS[2][1]+CHOSEN_POINTS[3][1])/2)

    bottom_middle = [xb,yb]
    upper_middle = [xu,yu]
    return bottom_middle, upper_middle
def click_event(event,x,y,flags,param):
    global CHOSEN_POINTS_COUNTER
    global CHOSEN_POINTS

    if event == cv2.EVENT_LBUTTONDOWN: # captures left button double-click
        CHOSEN_POINTS[CHOSEN_POINTS_COUNTER] = x, y
        CHOSEN_POINTS_COUNTER = CHOSEN_POINTS_COUNTER + 1
        print(CHOSEN_POINTS)

def write_instructions(CHOSEN_POINTS, CHOSEN_POINTS_COUNTER, image):
    if(CHOSEN_POINTS_COUNTER==0):
        text ="zaznacz lewy dolny rog pasow"
    elif(CHOSEN_POINTS_COUNTER==1):
        text ="zaznacz prawy dolny rog pasow"
    elif (CHOSEN_POINTS_COUNTER == 2):
        text ="zaznacz prawy gorny rog pasow"
    elif (CHOSEN_POINTS_COUNTER == 3):
        text = "zaznacz lewy gorny rog pasow"
    elif (CHOSEN_POINTS_COUNTER == 4):
        text = "zaznacz dolny horyzont"
    elif (CHOSEN_POINTS_COUNTER == 5):
        text = "zaznacz gorny horyzont"
    for i in range(CHOSEN_POINTS_COUNTER):
        cv2.circle(image, CHOSEN_POINTS[i], 2, [125, 246, 55], 5)
    cv2.putText(image,text,[(image.shape[0]//2), image.shape[1]//2],cv2.FONT_HERSHEY_DUPLEX,1,[125, 246, 55],thickness=2)
    return image
def camera_calibration(imgpath,txtpath):

    global CHOSEN_POINTS_COUNTER
    global CHOSEN_POINTS
    if os.path.isfile(txtpath):
        with open(txtpath) as f:
            lines = f.read().split('\n')
            lines = lines[0:-1]#usuniecie ostatniego entera
            for line in lines:
                line = line.split(",")
                CHOSEN_POINTS[CHOSEN_POINTS_COUNTER] = line
                CHOSEN_POINTS_COUNTER=CHOSEN_POINTS_COUNTER+1
        return CHOSEN_POINTS
    image = cv2.imread(imgpath)
    print(CHOSEN_POINTS)

    cv2.imshow("Camera calibrator", image)
    cv2.setMouseCallback('Camera calibrator', click_event)
    while CHOSEN_POINTS_COUNTER < 6:
        img_copy = image.copy()
        img_copy = write_instructions(CHOSEN_POINTS, CHOSEN_POINTS_COUNTER, img_copy)
        cv2.imshow("Camera calibrator", img_copy)
        cv2.waitKey(1)

    if CHOSEN_POINTS_COUNTER == 6:
        cv2.destroyAllWindows()
        number_of_segments = input("podaj ilosc zaznaczonych segmentow: ")
        CHOSEN_POINTS[CHOSEN_POINTS_COUNTER] = (number_of_segments,0)
        CHOSEN_POINTS_COUNTER = CHOSEN_POINTS_COUNTER + 1
        for i in range(len(CHOSEN_POINTS)):
            #print(i)
            with open(txtpath, 'a') as f:
                f.write(str(CHOSEN_POINTS[i][0])+","+str(CHOSEN_POINTS[i][1]))
                f.write("\n")
        return CHOSEN_POINTS



