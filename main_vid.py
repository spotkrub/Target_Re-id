import time
import numpy as np
import pandas as pd
import cv2 as cv

import yaml
import sensors
import reid_known as reid
import reid_known_plots as reid_plt

from torch.multiprocessing import Pool, Process, set_start_method, Lock, Queue

def get_frame_filename(f, media_dir, agent_name):
    num_digit = len(str(f))
    frame_name = str(f)
    if num_digit == 1:
        frame_name = '000' + frame_name
    elif num_digit == 2:
        frame_name = '00' + frame_name
    elif num_digit == 3:
        frame_name = '0' + frame_name    

    filename = media_dir + "imgseq/" + frame_name + "_" + agent_name + ".png"

    return filename

def draw_ooi_id(img, OOI_im_coords, OOI_ids, fontColor):    
    font                   = cv.FONT_HERSHEY_SIMPLEX
    fontScale              = 2
    thickness              = 3
    lineType               = 2

    OOI_im_coords = np.array(OOI_im_coords, dtype='int')
    for i, OOI_coord in enumerate(OOI_im_coords):
        if OOI_ids[i] == -1:
            OOI_id = 'X'
        else:
            OOI_id = str(int(OOI_ids[i]))
        text_coord = OOI_coord.copy()
        text_coord[0] -= 18
        text_coord[1] -= 15
        cv.circle(img, tuple(OOI_coord), 5, fontColor, -1)
        cv.putText( img, 
                    OOI_id, 
                    text_coord, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                    bottomLeftOrigin=False
                    )

def plot_reid(img, results_ooi):
    fontColor_overlap = (0, 255, 0)
    fontColor_non_overlap = (255, 0, 255)

    # Color OOI coordinates on image A
    overlap_idA = results_ooi[:, -1] >= 0
    non_overlap_idA = np.invert(overlap_idA)
    draw_ooi_id(img, results_ooi[non_overlap_idA, 7:9], results_ooi[non_overlap_idA, 13], fontColor_non_overlap)
    draw_ooi_id(img, results_ooi[overlap_idA, 7:9], results_ooi[overlap_idA, 13], fontColor_overlap)

def plot_frame(img, f):
    font                   = cv.FONT_HERSHEY_SIMPLEX
    fontScale              = 2
    fontColor              = (0, 0, 255)
    thickness              = 3
    lineType               = 2

    text_coord = np.array([30, 50])
    cv.putText( img, 
                str(f),
                text_coord, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType,
                bottomLeftOrigin=False
                )


def init_cam(id, media, media_dir):
    """ Load camera output data """

    with open(media_dir + 'agents.yml', 'r') as file:
        cam_info = yaml.safe_load(file)

    cam = sensors.cam(cam_info["agent" + str(id)])
    vid_file = media_dir + cam.cam_output[media]
    cap = cv.VideoCapture(vid_file)
    csv_file = media_dir + cam.cam_output['poses']
    cam_poses = pd.read_csv(csv_file, header = 0)

    """ Load camera detection model """
    with open('config_detection.yml', 'r') as file:
        detection_config = yaml.safe_load(file)
    
    detection_model = 'yolov5_setup1'
    detect_setup = detection_config[detection_model]
    cam.detect_init(detect_setup)

    return cam, cap, cam_poses


def agent(q_send, q_rcv, id, media, media_dir):
    """ Data Initialization """
    cam_send, cap, cam_poses = init_cam(id, media, media_dir)

    """ Data Acquisition """
    f = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Stream ended. Exiting ...")
            break

        # Update camera pose
        cam_send.update_pose_csv(cam_poses.loc[f])
        # reid.add_gaussian_noise(cam_send)

        """ Detection and Pre-processing """
        results_send = cam_send.detect_OOI(img)
        results_send = cam_send.check_detection_history(results_send)

        # Send
        q_send.put([cam_send, results_send])

        # Receive
        update = q_rcv.get()
        cam_rcv = update[0]
        results_rcv = update[1]
        results_reid_send, _, _ = reid.local_reid(cam_send, cam_rcv, results_send, results_rcv)

        # Draw Reid
        plot_reid(img, results_reid_send)
        plot_frame(img, f)

        # Display reid results at 1/4 of origial image 
        img = cv.resize(img, (800, 450))
        cv.imshow('Agent' + str(id), img)
        


        if cv.waitKey(1) == ord('q'):
            cv.waitKey(10000000)        

        if f == 0:
            cv.waitKey(10000000)

        # Frame count
        f += 1

    # Release everything if job is finished
    cap.release()
    cv.destroyAllWindows() 

def agent_csv_only(q_send, q_rcv, id, media):
    pass


def single_agent_tracking(id, media, media_dir):
    """ Data Initialization """
    cam_send, cap, cam_poses = init_cam(id, media, media_dir)

    """ Data Acquisition """
    f = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Stream ended. Exiting ...")
            break

        # Update camera pose
        cam_send.update_pose_csv(cam_poses.loc[f])

        """ Detection and Pre-processing """
        frame_detections = cam_send.detect_OOI(img)
        results_send = cam_send.check_detection_history(frame_detections)

        # Draw id
        results_send = np.concatenate((results_send, np.array([results_send[:, 0]]).T), axis=1)

        plot_reid(img, results_send)
        plot_frame(img, f)

        # Display reid results at 1/4 of origial image 
        img = cv.resize(img, (800, 450))
        cv.imshow('Agent' + str(id), img)
        if cv.waitKey(1) == ord('q'):
            break

        # cv.waitKey(50)

        f += 1

    # Release everything if job is finished
    cap.release()
    cv.destroyAllWindows()


def agent_single_frame(q_send, q_rcv, id, media, media_dir, f):
    """ Data Initialization """
    cam_send, _, cam_poses = init_cam(id, media, media_dir)

    """ Data Acquisition """
    frame_imgseq = get_frame_filename(f, media_dir, cam_send.name)
    cam_send.update_pose_csv(cam_poses.loc[f])
    reid.add_gaussian_noise(cam_send)

    """ Detection and Pre-processing """
    img = cv.imread(frame_imgseq)
    results_send = cam_send.detect_OOI(img)

    # Send
    q_send.put([cam_send, results_send])

    # Receive
    update = q_rcv.get()
    cam_rcv = update[0]
    results_rcv = update[1]

    results_reid_send, results_reid_rcv, transformed_OOIs = reid.local_reid(cam_send, cam_rcv, results_send, results_rcv)

    # Draw Reid
    plot_reid(img, results_reid_send)
    plot_frame(img, f)
    reid_plt.plot_results(cam_send, cam_rcv, results_reid_send, results_reid_rcv, transformed_OOIs)



    # Resize 1/4 of origial and display
    img = cv.resize(img, (800, 450))
    cv.imshow('Agent' + str(id), img)
    cv.waitKey(100000000)


def reid_2_videos(media, media_dir, agent_id0, agent_id1):
    q1 = Queue()
    q2 = Queue()

    p0 = Process(target = agent, args=(q1, q2, agent_id0, media, media_dir))
    p1 = Process(target = agent, args=(q2, q1, agent_id1, media, media_dir))

    p0.start()
    p1.start()
    p0.join()
    p1.join()


def reid_2_images(media, media_dir, agent_id0, agent_id1, f):
    q1 = Queue()
    q2 = Queue()

    p0 = Process(target = agent_single_frame, args=(q1, q2, agent_id0, media, media_dir, f))
    p1 = Process(target = agent_single_frame, args=(q2, q1, agent_id1, media, media_dir, f))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

def single_tracking(media, media_dir, agent_id):
    p0 = Process(target = single_agent_tracking, args=(agent_id, media, media_dir))
    p0.start()
    p0.join()



if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass


    """ Setup """
    media = "vid"
    media_dir = '../data/Blender/cam_move/' 
    # media_dir = '../data/Real/Set_01_1stgroup_6thpairdifferentangle/'
    # media_dir = '../data/Real/Set_02_10th_pair_best/'
    # media_dir = '../data/Real/Set_03_15th_pair/'
    # media_dir = '../data/Real/Set_04_cars/'

    agent_id0 = '0'
    agent_id1 = '1'
    f = 550 # For evaluation 1 only


    """ Evaluation """
    # reid_2_videos(media, media_dir, agent_id0, agent_id1) # Evaluation 0
    reid_2_images(media, media_dir, agent_id0, agent_id1, f) # Evaluation 1
    # single_tracking(media, media_dir, agent_id0) # Evaluation 2
