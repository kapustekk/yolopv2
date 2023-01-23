# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:56:10 2022

@author: ADM
"""
import cv2
import numpy as np
import math


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 90)

    #cv2.imshow("canny", canny)
    #cv2.waitKey(1)
    return canny


def oblicz_wspolczynniki(line):
    print(np.shape(line))
    x1, y1, x2, y2 = line.reshape(4)
    b = ((x2 * y1 - x1 * y2) / (x2 - x1))
    a = (y2 - b) / x2
    fi = math.atan(a) * 180 / math.pi
    return a, b, fi


def oblicz_dlugosc_linii(line):
    x1, y1, x2, y2 = line.reshape(4)
    dlugosc = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dlugosc


def okresl_kierunki(lines):
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            a, b, fi = oblicz_wspolczynniki(line)
            if fi < -25 and fi > -90:
                left_lines.append(line)
            if fi > 25 and fi < 90:
                right_lines.append(line)

    if (right_lines) is None:
        return left_lines, None
    elif (left_lines) is None:
        return None, right_lines
    else:
        return left_lines, right_lines


def display_hough_lines(image, lines):
    #line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(image, (x1,y1), (x2,y2), (0, 0, 255), 3)
    return image

def get_hough_lanes(frame):
    cannyimg = canny(frame)
    lines = cv2.HoughLinesP(cannyimg, 1, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=5)
    #left_lanes, righ_lanes = okresl_kierunki(lines)
    return lines

