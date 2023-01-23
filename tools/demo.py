import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
from PointsChoosing import camera_calibration, find_optic_middle
from ImageWrapping import warp_image_to_birdseye_view, warp_point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from matplotlib import pyplot as plt

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def separate_points(points, left_left_lane_points,left_lane_points, right_lane_points, right_right_lane_points, img_middle):
    left_distance_list = []
    right_distance_list =[]
    for point in points:
        distance = point[0] - img_middle
        if distance <0:
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

def find_middle_pixel_on_height(lane_mask,height):
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

def detect(cfg,opt,calibration_points):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    #cv2.setMouseCallback('image', select_point)
    #cv2.imshow('image', img)

    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):

        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()

        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # DRIVING AREA PREDICT
        #da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        #da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        #_, da_seg_mask = torch.max(da_seg_mask, 1)
        #da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)



        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = np.uint8(ll_seg_mask)
        # Lane line post-processing

        #ll_seg_mask = process_lane_mask(ll_seg_mask)
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)
        #color_area = np.zeros((ll_seg_mask.shape[0],ll_seg_mask.shape[1], 3), dtype=np.uint8)
        #color_area[ll_seg_mask == 1] = [255, 0, 0]

        img_det_copy = img_det.copy()
        birds_img, M, Minv = warp_image_to_birdseye_view(img_det_copy, calibration_points)
        birds_ll_seg_img, M_ll_seg, Minv_ll_seg = warp_image_to_birdseye_view(ll_seg_mask, calibration_points)
        img_det_birdseye=cv2.bitwise_and(birds_img,birds_img, mask=birds_ll_seg_img)


        optic_middle_bottom, optic_middle_upper = find_optic_middle(calibration_points)
        optic_middle_bottom_warp = warp_point(optic_middle_bottom,M)
        optic_middle_upper_warp = warp_point(optic_middle_upper,M)
        cv2.line(img_det, optic_middle_bottom, optic_middle_upper, [255, 150, 150], 3)

        cv2.line(img_det_birdseye, optic_middle_bottom_warp, optic_middle_upper_warp, [255,150,150], 3)

        points_list =[]
        left_lane_points = []
        right_lane_points = []
        left_left_lane_points = []
        right_right_lane_points = []
        points_density = 20
        bottom_horizon = calibration_points[4]
        upper_horizon = calibration_points[5] #gorny horyzont - pikselowo mniejsza wartość!
        upper_horizon_warped = warp_point(upper_horizon,M)
        bottom_horizon_warped = warp_point(bottom_horizon,M)
        print(upper_horizon_warped)
        cv2.circle(img_det_birdseye,bottom_horizon_warped, 2, [0,255,0],3)
        cv2.circle(img_det_birdseye,upper_horizon_warped, 2, [0,255,0],3)
        for i in range(points_density):
            D = bottom_horizon[1]-upper_horizon[1]
            horizontal_line=bottom_horizon[1]-(i*D//points_density)
            cv2.line(img_det, (0,horizontal_line), (ll_seg_mask.shape[1],horizontal_line),[0,0,100],1)
            points = find_middle_pixel_on_height(ll_seg_mask,horizontal_line)
            left_left_lane_points,left_lane_points,right_lane_points,right_right_lane_points = separate_points(points,left_left_lane_points,left_lane_points,right_lane_points,right_right_lane_points, ll_seg_mask.shape[1]//2)

            for point in left_left_lane_points:
                points_list.append(point)
                cv2.circle(img_det, point, 2, [0,150,150],3)
            previous_element = []
            for point in left_lane_points:
                if len(previous_element) != 0:
                    if abs(previous_element[0] - point[0]) < ll_seg_mask.shape[1] // 5:
                        cv2.line(img_det, previous_element, point, [242, 123, 52], 3)
                previous_element = point
                points_list.append(point)
                cv2.circle(img_det, point, 2, [0,255,0],3)

            for point in right_right_lane_points:
                points_list.append(point)
                cv2.circle(img_det, point, 2, [0,150,150],3)
            previous_element = []
            for point in right_lane_points:

                if len(previous_element) != 0:
                    if abs(previous_element[0] - point[0]) < ll_seg_mask.shape[1] // 5:
                        cv2.line(img_det, previous_element, point, [242, 123, 52], 3)
                previous_element = point
                points_list.append(point)
                cv2.circle(img_det, point, 2, [0,255,0],3)

        middle_middle=int(img_det.shape[1]/2), int(img_det.shape[0]/2)
        middle_lower=(int(img_det.shape[1]/2), img_det.shape[0])
        cv2.line(img_det, (middle_lower), (middle_middle), [10,20,255], 1)#linia srodkowa

        # img_det = img_det.astype(np.uint8)
        #img_det = cv2.resize(img_det, (640, 480), interpolation=cv2.INTER_LINEAR)
        #img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)


        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=[123,123,255], line_thickness=2)
                bottom_y = int(xyxy[3])
                mid_x = int((xyxy[0]+xyxy[2])/2)
                cv2.circle(img_det, (mid_x,bottom_y), 2, [0,0,255],3)

        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)

        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

        cv2.imshow("lanes", img_det)
        cv2.waitKey(1)

        cv2.imshow("birdseye", img_det_birdseye)
        cv2.waitKey(0)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    calibration_points = camera_calibration("inference/vid2/calibration.png", "inference/vid2/calibration.txt")
    print("DEMO!",calibration_points)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/single_image', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(cfg,opt,calibration_points)
