import numpy as np
import cv2
import math

def warp_point(point, M):
    x = point[0]
    y = point[1]
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    xM = int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d)
    yM = int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d)
    return [xM, yM]


def get_warp_perspective(calibration_points, x_conv,y_conv,number_of_segments):
    src, dst = get_warp_points(calibration_points, x_conv,y_conv,number_of_segments)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def warp_image_to_birdseye_view(image, M):
    height = image.shape[0]
    width = image.shape[1]
    warped_image = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped_image

def calculate_distance_between_points(point1, point2):
    x1 =(point1[0])
    y1 =(point1[1])
    x2=(point2[0])
    y2=(point2[1])
    x_dist = x2-x1
    y_dist = y2-y1

    return (x_dist,y_dist)

def estimate_real_distance(distance,x_conv,y_conv):
    #distance ma miec format (x_pix, y_pix) - odleglosc x y na obrazie birdseye
    x_pix = distance[0]
    y_pix = distance[1]
    real_distance_x = x_pix/x_conv#dystans w metrach
    real_distance_y = y_pix/y_conv
    return (real_distance_x,real_distance_y)
def get_warp_points(calibration_points,x_conv,y_conv,number_of_segments):
    # Save corner values for source and destination partitions
    corners = np.float32([calibration_points[0],calibration_points[1],calibration_points[2],calibration_points[3]])
    dst_height = 4*y_conv #height/60 to 1m dla 720p: 12pikseli - 1m
    dst_width = number_of_segments*0.5*x_conv
    dst0 = corners[0]
    dst1=(int(corners[0][0]+dst_width),int(corners[1][1]))
    dst2=(int(corners[0][0]+dst_width),int(corners[1][1]-dst_height))
    dst3=(int(corners[0][0]),int(corners[0][1]-dst_height))
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32([dst0,  dst1, dst2,dst3])

    return src_points, dst_points
