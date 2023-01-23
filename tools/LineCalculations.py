import cv2
import numpy as np
import math

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0, 0, 255), 3)
    return line_image

def oblicz_wspolczynniki(point1,point2):
    x1, y1=point1
    x2, y2=point2
    b = ((x2 * y1 - x1 * y2) / (x2 - x1))
    a = (y2 - b) / x2
    fi = math.atan(a) * 180 / math.pi
    return a, b, fi

def wydluz_linie(lines, height):
    wydluzone = []
    i = 0
    for line in lines:
        dlugosc_linii = height * 0.2  # długosc rysowanych linii w pikselach
        a, b, fi = oblicz_wspolczynniki(line)
        y1 = height * 0.9
        x1 = (y1 - b) / a
        y2 = y1 - dlugosc_linii
        x2 = (y2 - b) / a
        wydluzone.append(np.array([int(x1), int(y1), int(x2), int(y2)]))
        i = i + 1

    return wydluzone


def oblicz_dlugosc_linii(line):
    x1, y1, x2, y2 = line.reshape(4)
    dlugosc = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dlugosc