import cv2
import numpy as np
import os.path
from numpy import genfromtxt

import glob
from input import *


CHOSEN_POINTS_COUNTER = 0
CHOSEN_POINTS = np.zeros((7,2), np.int)



def find_chessboard():
    frameSize = (1280, 720)
    boardSize = (10, 6)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('*.jpg')
    i = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (10, 6), None)
        # If found, add object points, image points (after refining them)
        if ret:
            i += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (boardSize[0], boardSize[1]), corners2, ret)
    print("Dobre iteracje: ", i)

    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print(ret)
    print(np.array(mtx))
    print(dist)
    print(rvecs)
    print(tvecs)


def calibrate_image(img):

    #img = cv2.imread('95.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    dst = cv2.resize(dst,(int(1280), int(720)))
    #cv2.imshow('calibresult.png', dst)
    #cv2.imwrite('result.jpg', dst)
    #cv2.waitKey(1)
    return dst
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
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    print(CHOSEN_POINTS)

    image = calibrate_image(image)

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



