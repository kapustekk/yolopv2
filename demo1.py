import argparse
import time
from pathlib import Path
import cv2
import torch
import math
import numpy as np
import keyboard
import sys
from tools.PointsChoosing import camera_calibration, find_optic_middle
from tools.ImageWrapping import warp_image_to_birdseye_view, warp_point, get_warp_perspective, calculate_distance_between_points, estimate_real_distance

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages


def position_on_road(img_det,road_middle,left_line,right_line,x_conv,y_conv,M):
    vehicle_front = road_middle
    cv2.circle(img_det, vehicle_front, 2, [0, 0, 255], 3)
    left_line_first_x = -1
    right_line_first_x = -1
    for point in left_line:
        if abs(point[1] - vehicle_front[1]) < 3:
            left_line_first_x = point[0]
            cv2.circle(img_det, point, 2, [255, 0, 255], 5)
            break
    for point in right_line:
        if abs(point[1] - vehicle_front[1]) < 3:
            right_line_first_x = point[0]
            cv2.circle(img_det, point, 2, [255, 0, 255], 5)
            break
    if left_line_first_x != -1 and right_line_first_x != -1:
        lines_middle_x = (left_line_first_x + right_line_first_x) // 2
        lines_middle_point = (lines_middle_x, vehicle_front[1])
        cv2.circle(img_det, lines_middle_point, 2, [255, 0, 255], 10)
        dist_polozenie = calculate_distance_between_points(warp_point(lines_middle_point, M),
                                                           warp_point(vehicle_front, M))
        dist_szerokosc_pasa = calculate_distance_between_points(warp_point((left_line_first_x, vehicle_front[1]), M),
                                                                warp_point((right_line_first_x, vehicle_front[1]), M))
        real_polozenie = estimate_real_distance(dist_polozenie, x_conv, y_conv)
        real_szerokosc = estimate_real_distance(dist_szerokosc_pasa, x_conv, y_conv)

        if real_szerokosc[0] < 2 or real_szerokosc[0] > 5 or abs(real_polozenie[0]) > abs(real_szerokosc[0]/2):
            cv2.putText(img_det, "bledny odczyt linii!", (300, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1, [0, 0, 255], thickness=2
                        )
        elif real_polozenie[0] > 0.8:
            cv2.putText(img_det, str(round(real_polozenie[0], 2)) + "m od srodka, zmiana pasa na prawy", (300, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1, [125, 246, 55], thickness=2)

        elif real_polozenie[0] < -0.8:
            cv2.putText(img_det, str(round(real_polozenie[0], 2)) + "m od srodka, zmiana pasa na lewy ", (300, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1, [125, 246, 55], thickness=2)

        else:
            cv2.putText(img_det, str(round(real_polozenie[0], 2)) + "m od srodka", (300, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1, [125, 246, 55], thickness=2)

def estimate_speed_towards_car(vehicle_front, unique_cars_10, x_conv,y_conv, M, fps=30):
    unique_cars_speed = []
    if len(unique_cars_10)>0:
        vehicle_front = warp_point(vehicle_front,M)
        unique_cars1=average_points(unique_cars_10, -30, -16)
        unique_cars2=average_points(unique_cars_10, -15, -1)
        time_between_points = 15 / fps
        if len(unique_cars1)==len(unique_cars2):
            for car1, car2 in zip(unique_cars1,unique_cars2):
                px_dist1 = calculate_distance_between_points(warp_point(car1,M),vehicle_front)
                px_dist2 = calculate_distance_between_points(warp_point(car2,M),vehicle_front)
                real_dist1 = estimate_real_distance(px_dist1,x_conv,y_conv)
                real_dist2 = estimate_real_distance(px_dist2,x_conv,y_conv)
                speed = (real_dist1[1]-real_dist2[1])/time_between_points
                unique_cars_speed.append(speed)
    return unique_cars_speed

def average_points(unique_cars,start_point,end_point):
    unique_cars_average = []
    for car in unique_cars:
        car = car[start_point:end_point]
        sum_y1=0
        sum_x1=0
        for point in car:
             sum_y1 = sum_y1+point[1]
             sum_x1 = sum_x1+point[0]
        avg_y1 = int(sum_y1/len(car))
        avg_x1 = int(sum_x1/len(car))
        unique_cars_average.append((avg_x1,avg_y1))
    return unique_cars_average

def label_cars(set_of_found_cars, height, number_of_frames):
    unique_cars = []
    if len(set_of_found_cars)>number_of_frames:
        first_set = set_of_found_cars[-(number_of_frames+1)]
        for car_point in first_set:
            unique_car = []
            unique_car.append(car_point)
            for set in set_of_found_cars[-number_of_frames:-1]:
                get_point = True
                for car2 in set:
                    if abs(car2[0]-car_point[0])<height//60 and abs(car2[1]-car_point[1])<height//30 and get_point:
                        unique_car.append(car2)
                        car_point = car2
                        get_point=False
                if len(unique_car)==number_of_frames:
                    unique_cars.append(unique_car)
    return unique_cars


def separate_points(points, left_left_lane_points,left_lane_points, right_lane_points, right_right_lane_points, img_middle):
    left_distance_list = []
    right_distance_list =[]
    for point in points:
        distance = point[0] - img_middle
        if distance < 0:
            left_distance_list.append(abs(distance))
        if distance >= 0:
            right_distance_list.append(abs(distance))
    if len(left_distance_list)>0:
        min_left_idx = (left_distance_list.index((min(left_distance_list))))
        left_lane_points.append(points[min_left_idx])
        if len(left_distance_list)>1:
            left_left_lane_points.append(points[min_left_idx-1])

    if len(right_distance_list) > 0:
        min_right_idx = (right_distance_list.index(min(right_distance_list)))
        min_right_idx = min_right_idx + len(left_distance_list)
        right_lane_points.append(points[min_right_idx])
        if len(right_distance_list)>1:
            right_right_lane_points.append(points[min_right_idx+1])

    return left_left_lane_points, left_lane_points, right_lane_points, right_right_lane_points


def find_middle_pixel_on_height(lane_mask, height):
    horizontal_lane=(lane_mask[height])
    cnt0=0
    cnt1=0
    previous_pixel = 0
    points_list=[]
    for current_pixel in range(len(horizontal_lane)):
        #print(current_pixel)
        if horizontal_lane[previous_pixel]*horizontal_lane[current_pixel]==1:
            cnt1=cnt1+1
        elif horizontal_lane[previous_pixel]*horizontal_lane[current_pixel]==0 and cnt1!=0:
            points_list.append((current_pixel-cnt1//2,height))
            cnt1=0
        elif horizontal_lane[current_pixel]==0:
            cnt0=cnt0+1
        previous_pixel = current_pixel

    return points_list


def display_from_list(img, list_of_points, height, color):
    previous_element = []
    for element in list_of_points:
        # cv2.circle(img, element, 1, [0, 255, 255], 2)
        if len(previous_element) != 0:
            if abs(previous_element[0] - element[0]) < height // 5: # ll_seg_mask
                cv2.line(img, previous_element, element, color, 2)
        previous_element = element
    return img


def find_point_on_height(list_of_points, height):
    for point in list_of_points:
        if point[1] == height:
            return point


def average_point(list_of_points): # ew. wywalać tee liste, która ma [(1049, 737), None, None, None, None] tylko jeden/dwa? elementy
    average_value = 0
    i = 0
    for point in list_of_points:
        if point != None:
            i = i+1
            average_value = average_value + point[0]
    average_value = average_value/i
    wanted_point = (int(average_value), list_of_points[0][1])
    return wanted_point


def average_line(set_of_lines):
    b = 0
    c = 0
    work_list = []
    final_list = []
    for i in range(5):
        work_list.append(set_of_lines[-i - 1])
    for i in range(5):
        a = len(work_list[i])
        if a > b:
            b = a
            c = i
    longest_list = work_list[c]
    work_list.pop(c)
    for point in longest_list:
        all_points_on_height = [point]
        for other_list in work_list:
            all_points_on_height.append(find_point_on_height(other_list, point[1]))
        final_list.append(average_point(all_points_on_height))
    return final_list


def display_from_set(img, set_of_lines, mask):
    if len(set_of_lines) < 5:
        return img
    else:
        final_list = average_line(set_of_lines)
        img = display_from_list(img, final_list, mask, [0, 255, 0])
        return img


def check_points(lane_points, point, last_points, index, i):
    if len(lane_points) > 0 and point[1] == lane_points[-1][1]:
        a = math.sqrt(
            ((point[0] - lane_points[-2][0]) * (point[0] - lane_points[-2][0])) + (
                    (point[1] - lane_points[-2][1]) * (
                    point[1] - lane_points[-2][1])))
        b = math.sqrt(
            ((lane_points[-1][0] - lane_points[-2][0]) * (
                    lane_points[-1][0] - lane_points[-2][0])) + (
                    (lane_points[-1][1] - lane_points[-2][1]) * (
                    lane_points[-1][1] - lane_points[-2][1])))
        if b > a:
            lane_points.pop(-1)
            lane_points.append(point)
    elif len(lane_points) > 0 and last_points[index] == lane_points[-1]:
        lane_points.append(point)
    elif len(lane_points) > 4 and last_points[index][1] == lane_points[-1][1] and i < 15:
        a, b, c, d = 0, 0, 0, 0
        a = lane_points[-3][0]-lane_points[-4][0]
        b = lane_points[-2][0] - lane_points[-3][0]
        c = lane_points[-1][0] - lane_points[-2][0]
        d = last_points[index][0] - lane_points[-2][0]
        if a > 0 and b > 0:
            diff1 = abs(c-(a+b)/2)
            diff2 = abs(d-(a+b)/2)
            if diff1 > diff2:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
        elif a < 0 and b < 0:
            diff1 = abs(c-(a+b)/2)
            diff2 = abs(d-(a+b)/2)
            if diff1 > diff2:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
        elif a > 0 and b < 0:
            if last_points[index][0] < lane_points[-2][0] and abs(last_points[index][0]-lane_points[-2][0])<threshold:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
        elif a < 0 and b > 0:
            if last_points[index][0] > lane_points[-2][0] and abs(last_points[index][0]-lane_points[-2][0])<threshold:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
    return lane_points


def finding_closest_point_by_width_and_height(point, last_points, howfar, right_lane_points, left_lane_points, i):
    index = 0
    for otherpoint in last_points:  # znajdowanie najbliższego punktu z linii niżej dla danego point na wysokosci powyżej 5 linii
        howclose = math.sqrt(((point[0] - otherpoint[0]) * (point[0] - otherpoint[0])) + (
                    (point[1] - otherpoint[1]) * (point[1] - otherpoint[1])))
        # print("odleglosc punktu ", points.index(point)+1," od punktu ", last_points.index(otherpoint)+1, "=sqrt((", point[0], "-", otherpoint[0], ")^2 + (", point[1], "-",otherpoint[1], ")^2)=",howclose)
        if howclose < howfar and howclose < threshold:
            howfar = howclose
            index = last_points.index(otherpoint)  # punkt znaleziony i jego index
    if howfar != 100000000000:
        right_lane_points = check_points(right_lane_points, point, last_points, index, i)
        # if len(right_lane_points) > 0 and last_points[index] == right_lane_points[-1]:
        #     right_lane_points.append(point)
        left_lane_points = check_points(left_lane_points, point, last_points, index, i)

    return right_lane_points, left_lane_points


def checking_points_distance(list_of_points):
    for point in list_of_points:
        if point != list_of_points[-1]:
            if abs(point[0] - list_of_points[list_of_points.index(point) + 1][0]) > width_threshold:
                list_of_points.pop(list_of_points.index(point) + 1)
                checking_points_distance(list_of_points)
    return list_of_points


def deleting_far_points_from_list(list_of_points):
    inverted_list = list_of_points[::-1]
    inverted_list = checking_points_distance(inverted_list)
    list_of_points = inverted_list[::-1]
    return list_of_points


def appending_list_if_found_or_not(side_points_list, previous_lines_list):
    if len(side_points_list) > 6:
        previous_lines_list.append(side_points_list)
    if len(side_points_list) <= 6 and len(previous_lines_list) > 0:
        side_points_list = previous_lines_list[-1]
        previous_lines_list.append(side_points_list)
    return previous_lines_list


def furthest_points(line):
    xx = []
    for point in line:
        xx.append(point[0])
    max_left, max_right = min(xx), max(xx)
    return [max_left, max_right]


def aproximate_line(set_of_lines, degree, fleft, fright, horizon1, horizon2):
    iksy, igreki, poly_line = [], [], []
    if len(set_of_lines) > 5:
        avg_line = average_line(set_of_lines)
        for point in avg_line:
            iksy.append(point[0])
            igreki.append(point[1])
        polynomial = np.poly1d(np.polyfit(iksy, igreki, degree))
        a = furthest_points(avg_line)
        if fleft > a[0]:
            fleft = a[0]
        if fright < a[1]:
            fright = a[1]
        ttt = np.linspace(fleft, fright, fright - fleft + 1)  # szerokośc czyli najdalej lewa i prawa
        for x in ttt:
            if horizon2 >= int(polynomial(x)) >= horizon1:
                poly_line.append([int(x), int(polynomial(x))]) # warunek upper bottom horyzont
    return poly_line, fleft, fright


divide = 40
points_density = 20  # dobre 20/40
first_phase = points_density//2
second_phase = first_phase+1

def make_parser(test_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=test_path, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser


def detect(calibration_points):
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    #add start
    # Getting image/video resolution
    dataset.__iter__()
    (path, img, img_det, vid_cap) = dataset.__next__()
    height, width, _ = img_det.shape
    global width_threshold, threshold
    width_threshold = width // divide
    threshold = math.sqrt(width*height)//divide

    if len(calibration_points)>0:
        bottom_horizon = calibration_points[4]
        upper_horizon = calibration_points[5]  # gorny horyzont - pikselowo mniejsza wartość!
        D = bottom_horizon[1] - upper_horizon[1]
        optic_middle_bottom, optic_middle_upper = find_optic_middle(calibration_points)
        optic_middle=(((optic_middle_bottom[0]+optic_middle_upper[0])//2),((optic_middle_bottom[1]+optic_middle_upper[1])//2))
        vehicle_front = (optic_middle[0],bottom_horizon[1])
        img_middle = optic_middle[0]
        number_of_segments = calibration_points[6][0]

        y_conv = int(height / 60)  # height/60 to 1m dla 720p: 12pikseli po y = 1m
        x_conv = int(height / 15)  # height/15 to 1 m, dla 720p 3piksele po x = 1m
        M, Minv = get_warp_perspective(calibration_points, x_conv, y_conv,number_of_segments)
        # optic_middle_upper_warp = warp_point(optic_middle_upper, M)
    else:
        img_middle = width//2
        print(img_middle)
        bottom_horizon = (img_middle,height-5)
        upper_horizon = (img_middle,int(0.6*height))  # gorny horyzont - pikselowo mniejsza wartość!
        D = bottom_horizon[1] - upper_horizon[1]
        

    set_of_lines_right = []
    set_of_lines_left = []
    set_of_found_cars = []

    fleft1, fright1 = width*2, 0
    fleft2, fright2 = width*2, 0
    #add end
    
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, img_det, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            h, w, _ = img_det.shape
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img_det.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            points_list = []
            left_lane_points = []
            right_lane_points = []
            left_left_lane_points = []
            right_right_lane_points = []
            last_points = []

            for i in range(points_density):
                horizontal_line = bottom_horizon[1] - (i * D // points_density)
                points = find_middle_pixel_on_height(ll_seg_mask, horizontal_line)
                if i <= first_phase:
                    left_left_lane_points, left_lane_points, right_lane_points, right_right_lane_points = separate_points(
                        points, left_left_lane_points, left_lane_points, right_lane_points, right_right_lane_points,
                        img_middle)
                    left_lane_points = deleting_far_points_from_list(left_lane_points)
                    right_lane_points = deleting_far_points_from_list(right_lane_points)

                for point in points:
                    # cv2.circle(img_det, point, 1, [255, 255, 255], 2)
                    howfar = 100000000000
                    points_list.append(point)
                    if i >= second_phase:
                        right_lane_points, left_lane_points = finding_closest_point_by_width_and_height(point,
                                                                                                        last_points,
                                                                                                        howfar,
                                                                                                        right_lane_points,
                                                                                                        left_lane_points,
                                                                                                        i)

                last_points = points

            set_of_lines_left = appending_list_if_found_or_not(left_lane_points, set_of_lines_left)
            set_of_lines_right = appending_list_if_found_or_not(right_lane_points, set_of_lines_right)
            left_poly_degree = 1
            right_poly_degree = 1
            if len(left_lane_points) > 10:
                left_poly_degree = 2
            if len(right_lane_points) > 10:
                right_poly_degree = 2

            left_line, fleft1, fright1 = aproximate_line(set_of_lines_left, left_poly_degree, fleft1, fright1,
                                                         upper_horizon[1], bottom_horizon[1])
            right_line, fleft2, fright2 = aproximate_line(set_of_lines_right, right_poly_degree, fleft2, fright2,
                                                          upper_horizon[1], bottom_horizon[1])
            img_det = display_from_list(img_det, left_line, h, (0, 255, 255))
            img_det = display_from_list(img_det, right_line, h, (0, 255, 255))
            #im0 = im0.astype(np.uint8)

            found_cars_points = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #add start

                    plot_one_box(xyxy, img_det, color=[123, 123, 255], line_thickness=2)
                    bottom_y = int(xyxy[3])
                    mid_x = int((xyxy[0] + xyxy[2]) / 2)
                    bottom_middle_point = (mid_x, bottom_y)
                    found_cars_points.append(bottom_middle_point)
                    #add end
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, img_det, line_thickness=3)

                if len(calibration_points) > 0:
                    # birds_img = warp_image_to_birdseye_view(img_det_copy, M)
                    # odleglosc od samochodu
                    set_of_found_cars.append(found_cars_points)
                    unique_cars_30 = label_cars(set_of_found_cars, h, 30)
                    unique_cars_5 = label_cars(set_of_found_cars, h,
                                               5)  # zrobic sredni punkt samochodu z 5 klatek i porownwyac w danej klatce i kolejnej do predkosci
                    average_cars_points = average_points(unique_cars_5, -5, -1)
                    estimated_speed_list = estimate_speed_towards_car(vehicle_front, unique_cars_30, x_conv, y_conv, M)

                    position_on_road(img_det, optic_middle_bottom, left_line, right_line, x_conv, y_conv,
                                     M)  # polozenie na pasie

                    i = 0
                    for point in average_cars_points:
                        # cv2.circle(birds_img, warp_point(point, M), 2, [0, 0, 255], 5)
                        px_distance = calculate_distance_between_points(warp_point(point, M),
                                                                        warp_point(vehicle_front, M))
                        real_dist = estimate_real_distance(px_distance, x_conv, y_conv)
                        diagonal_distnace = math.sqrt((real_dist[0] ** 2) + (real_dist[1] ** 2))
                        distance_written = False
                        if len(estimated_speed_list) > i:
                            speed_towards_car = 3.6 * estimated_speed_list[i]
                            cv2.putText(img_det, (str(round(speed_towards_car, 1)) + "km/h"),
                                        (point[0] - 30, point[1] + 30), cv2.FONT_HERSHEY_DUPLEX,
                                        1, [125, 246, 55], thickness=1)
                            if abs(real_dist[0]) < 0.5 and real_dist[1] < 15 and speed_towards_car > 30:
                                cv2.putText(img_det, ("!!!" + str(round(diagonal_distnace, 1)) + "m!!!"),
                                            (point[0] - 30, point[1]), cv2.FONT_HERSHEY_DUPLEX,
                                            1, [0, 0, 255], thickness=2)
                                distance_written = True
                        if distance_written == False:
                            cv2.putText(img_det, (str(round(diagonal_distnace, 1)) + "m"), (point[0] - 30, point[1]),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        1, [125, 246, 55], thickness=1)
                        i = i + 1
            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            #show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, img_det)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = img_det.shape[1], img_det.shape[0]
                        else:  # stream
                            fps, w, h = 30, img_det.shape[1], img_det.shape[0]
                            save_path += '.mp4'

                        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                            vid_writer.release()
                            cv2.destroyAllWindows()
                            sys.exit()

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_det)

                    cv2.imshow("lanes", img_det)
                    cv2.waitKey(1)

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    print(torch.cuda.is_available())
    test_path = 'inference/vid2'
    calibration_points = []
    calibrate = 1
    if calibrate == 1:
        calibration_points = camera_calibration(test_path+"/calibration.png", test_path+"/calibration.txt")

    opt =  make_parser(str(test_path)).parse_args()
    print(opt)

    with torch.no_grad():
            detect(calibration_points)
