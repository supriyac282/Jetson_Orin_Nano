#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import yaml
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import time

image_nets = [None, None]  # to store images for each camera
lock = [Lock(), Lock()]  # to synchronize access to image_nets
run_signals = [False, False]  # signals to trigger inference for each camera
detections = [[], []]  # to store detections for each camera
exit_signal = False


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2])  # * im_shape[1]
    x_max = (xywh[0] + 0.5 * xywh[2])  # * im_shape[1]
    y_min = (xywh[1] - 0.5 * xywh[3])  # * im_shape[0]
    y_max = (xywh[1] + 0.5 * xywh[3])  # * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output

def torch_thread(weights, img_size, camera_id, conf_thres=0.2, iou_thres=0.45):
    global image_nets, exit_signal, run_signals, detections

    print("Intializing Network for Camera", camera_id)

    model = YOLO(weights)

    while not exit_signal:
        start_time = time.time()  # Start time for measuring inference time
        if run_signals[camera_id - 1]:
            lock[camera_id - 1].acquire()
            img = cv2.cvtColor(image_nets[camera_id - 1], cv2.COLOR_BGRA2RGB)
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes
            detections[camera_id - 1] = detections_to_custom_box(det, image_nets[camera_id - 1])
            lock[camera_id - 1].release()
            end_time = time.time()  # End time for measuring inference time
            inference_time = end_time - start_time  # Calculate inference time for the current frame
            fps = 1.0 / inference_time  # Calculate frames per second
            print(f"Inference FPS for Camera {camera_id}: {fps:.2f}")
            run_signals[camera_id - 1] = False
        
        sleep(0.01)


def main():
    global image_nets, exit_signal, run_signals, detections

    capture_threads = [
        Thread(target=torch_thread, args=(opt.weights[i], opt.img_size, i + 1, opt.conf_thres)) for i in range(2)
    ]
    for thread in capture_threads:
        thread.start()
        
    with open('/usr/local/zed/samples/object detection/custom detector/python/pytorch_yolov8/coco.yaml', 'r') as f: 
    	data = yaml.safe_load(f)

    # Extract class names
    class_names = [data['names'][i] for i in range(len(data['names']))]
   

    print("Initializing Cameras...")

    # Initialize ZED cameras
    
    zed = [[] ,[]]
    
    for i in range(2):
        zed[i] = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 100
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()


    for i in range(2):
        status = zed[i].open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open camera {i + 1}")
            exit(-1)      
		
    image_left_tmp = [sl.Mat(), sl.Mat()] 
	
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    
    obj_param = sl.ObjectDetectionParameters()
    

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()    
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True	
    objects = [sl.Objects(), sl.Objects()] 
    
    for i in range(2):   
       zed[i].enable_positional_tracking(positional_tracking_parameters)
       zed[i].enable_object_detection(obj_param)

    # Display
    
    camera_infos = []
    camera_resolutions = []
    
    
    for i in range(2):
        camera_infos.append(zed[i].get_camera_information())
        camera_resolutions.append(camera_infos[i].camera_configuration.resolution)
    
    global_images = [[] ,[]]   

    point_cloud_resolutions = []
    point_clouds = []
    display_resolutions = []
    image_scales = []
    image_left_ocvs = []
    camera_configs = []
    tracks_resolutions = []
    track_view_generators = []
    image_track_ocvs = []
    
    viewer = gl.GLViewer()
    point_cloud_render = [sl.Mat(), sl.Mat()] 
    depth_map = [sl.Mat(), sl.Mat()] 

    
    for i in range(2):

        point_cloud_resolutions.append(sl.Resolution(min(camera_resolutions[i].width, 720), min(camera_resolutions[i].height, 404)))
        point_clouds.append(sl.Mat(point_cloud_resolutions[i].width, point_cloud_resolutions[i].height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU))
        viewer.init(camera_infos[i].camera_model, point_cloud_resolutions[i], obj_param.enable_tracking)
        display_resolutions.append(sl.Resolution(min(camera_resolutions[i].width, 1280), min(camera_resolutions[i].height, 720)))
        image_scales.append([display_resolutions[i].width / camera_resolutions[i].width, display_resolutions[i].height / camera_resolutions[i].height])
        image_left_ocvs.append(np.full((display_resolutions[i].height, display_resolutions[i].width, 4), [245, 239, 239, 255], np.uint8))
        camera_configs.append(camera_infos[i].camera_configuration)
        tracks_resolutions.append(sl.Resolution(400, display_resolutions[i].height))
        track_view_generators.append(cv_viewer.TrackingViewer(tracks_resolutions[i], camera_configs[i].fps, init_params.depth_maximum_distance))
        track_view_generators[i].set_camera_calibration(camera_configs[i].calibration_parameters)
        image_track_ocvs.append(np.zeros((tracks_resolutions[i].height, tracks_resolutions[i].width, 4), np.uint8))


    cam_w_pose = sl.Pose()
    print("Initialized Cameras")
    
    
    i = 0
    while viewer.is_available() and not exit_signal:

            # Grab frames from cameras
            if zed[i].grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

                lock[i].acquire()
                zed[i].retrieve_image(image_left_tmp[i], sl.VIEW.LEFT, sl.MEM.CPU, display_resolutions[i])
                image_nets[i] = image_left_tmp[i].get_data()

                lock[i].release()
                run_signals[i] = True

                while run_signals[i]:
                    sleep(0.001)

                lock[i].acquire()
                # Ingest detections from respective camera image net
                zed[i].ingest_custom_box_objects(detections[i])
                det_list = detections[i]

                lock[i].release()
                zed[i].retrieve_objects(objects[i], obj_runtime_param)

                # Retrieve display data
                zed[i].retrieve_measure(depth_map[i], sl.MEASURE.DEPTH)
                zed[i].retrieve_measure(point_clouds[i], sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_resolutions[i])
                point_clouds[i].copy_to(point_cloud_render[i])
                zed[i].get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
               

                # 3D rendering
                viewer.updateData(point_cloud_render[i], objects[i])

                # 2D rendering
                np.copyto(image_left_ocvs[i], image_left_tmp[i].get_data())
                cv_viewer.render_2D(image_left_ocvs[i], image_scales[i], objects[i], obj_param.enable_tracking)
                global_images[i] = cv2.hconcat([image_left_ocvs[i], image_track_ocvs[i]])

                # Tracking view
                track_view_generators[i].generate_view(objects[i], cam_w_pose, image_track_ocvs[i], objects[i].is_tracked)

                # Count of Objects
                count = len(det_list)
                cv2.putText(global_images[i], f"Detections: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Calculate Distance
                
                for det in det_list:	

                    bbox = det.bounding_box_2d
                    labelName = str(det.label)
                    labelIndex = int(labelName)
		    
                    center = np.mean(bbox, axis=0)
                    x = round(center[0])
                    y = round(center[1])

		    
                    err, depth_value = depth_map[i].get_value(x,y)
                    
                    cv2.putText(global_images[i], f"{class_names[labelIndex]} , Distance from Camera : {depth_value:.2f} meters ", (x,y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 1 , cv2.LINE_AA)
                    
                # Display cameras
                cv2.imshow(f"ZED Camera {i + 1}| 2D View and Birds View", global_images[i])

                key = cv2.waitKey(10)
                if key == 27:
                    exit_signal = True
            else:
               exit_signal = True
               
            i = 1 - i

    viewer.exit()
    exit_signal = True
    zed[0].close()
    zed[1].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['yolov8n.pt', 'yolov8l.pt'], help='model.pt paths for each camera')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()


    with torch.no_grad():
        main()
