import cv2 as cv
from time import time

import yaml
import sensors
import reid_known as reid
import reid_known_plots as reid_plt

import numpy as np

if __name__ == '__main__':
    # User Settings
    media = 'img_human' # ball or human

    cam_static_dir = '../data/Blender/cam_static/'

    
    with open(cam_static_dir + 'cam_static_info.yml', 'r') as file:
        cam_info = yaml.safe_load(file)
    
    with open('config_detection.yml', 'r') as file:
        detection_config = yaml.safe_load(file)
    detect_setup = detection_config['yolov5_setup1']

    camA = sensors.cam(cam_info["cam_static" + str(1)])
    camB = sensors.cam(cam_info["cam_static" + str(0)])

    # Initialise OOI detection model
    camA.detect_init(detect_setup)
    camB.detect_init(detect_setup)

    """ Data Acquisition """
    # Images captured by cameras
    imgA = cv.imread(cam_static_dir + camA.cam_output[media])
    imgB = cv.imread(cam_static_dir + camB.cam_output[media])

    """  Noise """
    # # Add Noise
    reid.add_gaussian_noise(camA)
    reid.add_gaussian_noise(camB) 

    # Remove Noise
    # reid.remove_gaussian_noise(camA)
    # reid.remove_gaussian_noise(camB)


    """ Detection and Pre-processing """
    results_ooiA = camA.detect_OOI(imgA) # OOIs Detection for camA
    results_ooiB = camB.detect_OOI(imgB) # OOIs Detection for camB


    """ Association """
    t2 = time() # Reid start time
    results_ooiA_reid, results_ooiB_reid, transformed_OOIs = reid.local_reid(camA, camB, results_ooiA, results_ooiB)
    t3 = time() # Reid end time
    print('ReID Time: '+ str(t3 - t2))

    """ Debug Plots """
    reid_plt.plot_original_id(imgA, imgB, results_ooiA, results_ooiB)
    reid_plt.plot_reid(imgA, imgB, results_ooiA_reid, results_ooiB_reid)
    reid_plt.plot_results(camA, camB, results_ooiA_reid, results_ooiB_reid, transformed_OOIs)