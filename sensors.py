import math as m
import numpy as np
from time import time
import cv2 as cv

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
import torch # Yolov5
# from ultralytics import YOLO # Yolov8

class cam:
    def __init__(self, cam_params):
        self.info = cam_params

        # Camera Intrinsics
        self.alpha_u, self.alpha_v = self.int_focal_len()     # Focal length [px]
        self.cx, self.cy = self.int_principal_pt()  # Principal point [px]
        self.s = cam_params['axis_skew']            # Axis skew

        # Camera Extrinsics
        self.roll = self.info['pose']['roll0']       # Roll [deg]
        self.pitch = self.info['pose']['pitch0']  #+ 180     # Pitch [deg]
        self.yaw = self.info['pose']['yaw0']         # Yaw [deg]
        self.worldx = self.info['pose']['worldx0']   # x [m]
        self.worldy = self.info['pose']['worldy0']   # y [m]
        self.worldz = self.info['pose']['worldz0']   # z [m]

        # Others
        self.id = cam_params['cam_id']
        self.name = cam_params['name']
        self.cam_output = cam_params['cam_output']

    def cam_mm_to_pixel(self):
        pixel_aspect_ratio = self.info['px_aspect_y'] / self.info['px_aspect_x']

        if self.info['sensor_fit'] == 'VERTICAL':
            sensor_size_mm = self.info['sensor_height']
            view_fac_px = pixel_aspect_ratio * self.info['res_y_px']
        else:
            sensor_size_mm = self.info['sensor_width']
            view_fac_px = self.info['res_x_px']

        x_pixel_size_mm_per_px = sensor_size_mm / view_fac_px
        y_pixel_size_mm_per_px = sensor_size_mm / view_fac_px * pixel_aspect_ratio

        return x_pixel_size_mm_per_px, y_pixel_size_mm_per_px

    def int_focal_len(self):
        x_pixel_size_mm_per_px, y_pixel_size_mm_per_px = self.cam_mm_to_pixel()
        alpha_u = self.info['focal_length'] / x_pixel_size_mm_per_px
        alpha_v = self.info['focal_length'] / y_pixel_size_mm_per_px
        return alpha_u, alpha_v

    def int_principal_pt(self):
        cx = self.info['res_x_px']/2
        cy = self.info['res_y_px']/2
        return cx, cy

    def get_rot_mat(self):
        roll_rad = m.radians(self.roll)
        pitch_rad = m.radians(self.pitch)
        yaw_rad = m.radians(self.yaw)
        
        # Roll
        Rx = np.matrix([[ 1,                        0,                  0        ],
                        [ 0,                m.cos(roll_rad),    -m.sin(roll_rad) ],
                        [ 0,                m.sin(roll_rad),     m.cos(roll_rad) ]])
        
        # Pitch
        Ry = np.matrix([[ m.cos(pitch_rad),         0,           m.sin(pitch_rad)],
                        [       0,                  1,                  0        ],
                        [-m.sin(pitch_rad),         0,           m.cos(pitch_rad)]])

        # Yaw
        Rz = np.matrix([[ m.cos(yaw_rad),  -m.sin(yaw_rad),             0        ],
                        [ m.sin(yaw_rad),   m.cos(yaw_rad),             0        ],
                        [       0,                  0,                  1        ]])

        rot_mat = Rz @ Ry @ Rx
        return rot_mat

    def get_t_mat(self):
        t_mat = np.matrix([[self.worldx],
                           [self.worldy],
                           [self.worldz]])
        return t_mat

    def get_int_mat(self):
        int_mat = np.matrix([[self.alpha_u, self.s,         self.cx],
                             [    0,        self.alpha_v,   self.cy],
                             [    0,           0,               1  ]])
        return int_mat
    
    def get_ext_mat(self):
        rot_mat = self.get_rot_mat()
        t_mat = self.get_t_mat()
        ext_mat = np.concatenate([rot_mat, t_mat], axis=1)
        return ext_mat

    def get_ext_mat_homogeneous(self):
        ext_mat = self.get_ext_mat()
        ext_mat_homogeneous = np.concatenate([ext_mat, np.zeros([1, 4])], axis=0)
        ext_mat_homogeneous[-1, -1] = 1
        return ext_mat_homogeneous

    def get_H_mat_1(self):
        int_mat = self.get_int_mat()
        ext_mat = self.get_ext_mat()
        ext_mat_truncate = np.delete(ext_mat, 2, 1)
        H_mat = int_mat @ ext_mat_truncate
        H_mat /= H_mat[2, 2]
        return H_mat

    def update_pose_csv(self, pose_df):
        self.roll =  pose_df.loc['roll']
        self.pitch =  pose_df.loc['pitch']
        self.yaw =  pose_df.loc['yaw']
        self.worldx =  pose_df.loc['x']
        self.worldy =  pose_df.loc['y']
        self.worldz =  pose_df.loc['z']

    def detect_init(self, detect_setup):
        # Update camera object
        self.detect_name = detect_setup['name']
        self.detect_weights = detect_setup['weights']
        self.detect_model = torch.hub.load(detect_setup['src_dir'], 'custom', path=self.detect_weights, source='local')
        self.detect_classes = detect_setup['classes']

    def detect_OOI(self, img):
        if self.detect_name == 'yolov5':
            OOI_results = self.detection_OOI_yolov5(img)

        # Processed output
        # Returns
        # ======================
        #   COLUMN  |   MEANING     
        # ======================
        # 0:    LOCAL ID
        # 1:    BB_centroidX
        # 2:    BB_centroidY
        # 3:    BB_WIDTH
        # 4:    BB_HEIGHT
        # 5:    CONFIDENCE
        # 6:    CLASS
        # 7:    BB_floorX
        # 8:    BB_floorY
        # 9:    WORLD_X
        # 10:   WORLD_Y
        # 11:   WORLD_Z
        # 12:   IN OTHER AGENT FOV
        # 13:   GLOBAL ID -- USE LOCAL ID FIRST
        # 14:   TOTAL FRAMES TRACKED
        # 15:   FRAMES LOST

        # Column 0 to 11
        original_id = np.array([range(len(OOI_results))])
        np_ones = np.ones((len(OOI_results), 1), dtype = int)
        np_zeros = np.zeros((len(OOI_results), 1), dtype = int)
        OOI_results = np.concatenate((original_id.T, OOI_results, np_ones, original_id.T, np_ones, np_zeros), axis=1)
        
        return OOI_results.astype(float)

    def detection_remove_classes(self, img_results_xywhcc_raw):
        img_results_xywhcc = []
        for detection in img_results_xywhcc_raw:
            if detection[-1] in self.detect_classes:
                img_results_xywhcc.append(detection)

        return np.array(img_results_xywhcc)
    
    def detection_draw_bb(self, img, OOI_results):
        for OOI_info in OOI_results:
            w = int(OOI_info[2])
            h = int(OOI_info[3])
            x = int(OOI_info[0] - w/2)
            y = int(OOI_info[1] - h/2)         
            cv.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 4)

    def detection_OOI_yolov5(self, img):
        # Detection start time
        t0 = time()

        # Detection
        img_results = self.detect_model(img)
        img_results_xywhcc_raw = np.array(img_results.pandas().xywh[0])[:, :6]  # Get ouput from yolov5

        # Detection end time
        t1 = time()
        # print('YOLO Inference Time: '+ str(t1 - t0))

        # Add in irrelevant class removal function here
        img_results_xywhcc = self.detection_remove_classes(img_results_xywhcc_raw)

        # Get centroid coordinates of bounding box
        OOI_im_coord = img_results_xywhcc[:, 0:2].copy()

        # Bottom of bounding box
        OOI_im_coord[:, 1] += img_results_xywhcc[:, 3] / 2

        # Add centroid (bottom) coordinates to end results of yolo
        OOI_world_coord = self.im_to_world(OOI_im_coord.astype(float))

        OOI_results = np.concatenate((img_results_xywhcc, OOI_im_coord, OOI_world_coord), axis=1)
        self.detection_draw_bb(img, OOI_results)
        
        return OOI_results

    def im_to_euc(self, OOI_im_coord):
        OOI_euc_coord = OOI_im_coord.copy()
        OOI_euc_coord[:, 1] = self.info['res_y_px'] - OOI_euc_coord[:, 1]
        return OOI_euc_coord

    def im_to_world(self, OOI_im_coord):
        # Convert OOIs from image coordinate system to euclidean coordinate system i.e y-axis points up now
        OOI_euc_coord = self.im_to_euc(OOI_im_coord)
        num_of_OOIs = len(OOI_euc_coord)

        # Distance and angle between OOI pixels to focal point
        fx = self.info['focal_length']
        fy = self.info['focal_length']
        x_pixel_size_mm_per_px, y_pixel_size_mm_per_px = self.cam_mm_to_pixel()
        dist_x = np.subtract(OOI_euc_coord[:, 0], self.info['res_x_px'] / 2) * x_pixel_size_mm_per_px # Horizontal distance
        dist_y = np.subtract(self.info['res_y_px']/2, OOI_euc_coord[:, 1]) * y_pixel_size_mm_per_px # Vertical distance
        alpha = np.arctan(np.divide(dist_x, fx)) # Horizontal angle
        beta =  np.arctan(np.divide(dist_y, fy)) # Vertical angle

        # Projection length between focal point and OOIs projected onto world coordinates
        pitch_blender = 90 - self.pitch # Adjust pitch from Blender's camera coordinate system
        pitch_rad = np.deg2rad(pitch_blender)

        # Calculate relative distance of OOIs from focal point
        delta_x = np.divide(self.worldz, np.sin(pitch_rad + beta))
        delta_x = np.multiply(np.multiply(delta_x,  np.cos(beta)), np.tan(alpha))
        delta_y = np.divide(self.worldz, np.tan(pitch_rad + beta))
        delta_z =  - np.ones(num_of_OOIs) * self.worldz
        delta_OOI = np.vstack((delta_x, delta_y, delta_z)).T
        
        # Account relative distance changes with rotation of camera's FOV 
        rot = Rotation.from_euler('xyz', (self.roll, pitch_rad, self.yaw), degrees=True)
        delta_OOI = rot.apply(delta_OOI)
        
        # Calculate world coordinates of OOIs
        OOI_world_coord = np.zeros((num_of_OOIs, 3))
        OOI_world_coord[:, :] = (self.worldx, self.worldy, self.worldz)        
        OOI_world_coord += delta_OOI
        
        return OOI_world_coord
    
    def nearest_neighbor_query_og(src, dst):
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)

        return distances.ravel(), indices.ravel()

    def createTrackerByName(trackerType):
        trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        # Create a tracker based on tracker name
        if trackerType == trackerTypes[0]:
            tracker = cv.legacy.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
            tracker = cv.legacy.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv.legacy.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv.legacy.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv.legacy.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv.legacy.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv.legacy.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
            tracker = cv.legacy.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)
        return tracker

    def add_to_tracker(self, img, bboxes):
        trackerType = "CSRT"    
        multiTracker = cv.legacy.MultiTracker_create()

        for bbox in bboxes:
            multiTracker.add(self.createTrackerByName(trackerType), img, bbox)

        return multiTracker

    def update_tracker(self, img, multiTracker):
        success, boxes = multiTracker.update(img)
        return success, boxes

    def release_tracker(multiTracker):
        success, boxes = multiTracker.release()
        return success, boxes


    def check_detection_history(self, new_detection_results):
        if hasattr(self, "detection_history"):
            num_tracked = np.shape(self.detection_history)[0]
            num_new = np.shape(new_detection_results)[0]

            if num_tracked < num_new:
                """ More OOI in new than tracked """
                src = new_detection_results[:, 1:3]
                dst = self.detection_history[:, 1:3]

                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(src)
                distances, indices = neigh.kneighbors(dst, return_distance=True)
                indices = indices.ravel()

                # Key part
                tracked_detection_results = self.detection_history.copy()
                new_ids = list(range(num_new))
                remaining_ids = [i for i in new_ids if i not in indices]

                # First update tracked ids
                for i in range(len(indices)):
                    tracked_detection_results[i, :] = new_detection_results[indices[i], :]
                    tracked_detection_results[i, 0] = i
                    tracked_detection_results[i, 13] = i

                # Then add in the new ids
                for j in remaining_ids:
                    tracked_detection_results = np.concatenate([tracked_detection_results, np.array([new_detection_results[j,:]])], axis=0)
                    tracked_detection_results[-1, 0] = num_new
                    tracked_detection_results[-1, 13] = num_new
                    num_new += 1
  
            else:
                """ More or equal OOI in tracked than new """
                src = self.detection_history[:, 1:3]                
                dst = new_detection_results[:, 1:3]

                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(src)
                distances, indices = neigh.kneighbors(dst, return_distance=True)
                indices = indices.ravel()

                # First update tracked ids
                tracked_detection_results = self.detection_history.copy()
                for i in range(len(indices)):
                    tracked_detection_results[indices[i], :] = new_detection_results[i, :]
                    tracked_detection_results[indices[i], 0] = indices[i]
                    tracked_detection_results[indices[i], 13] = indices[i]

                    # Update total frames tracked
                    tracked_detection_results[indices[i], 14] += 1

                tracked_ids = list(range(num_tracked))
                remaining_ids = [i for i in tracked_ids if i not in indices]

                # Increase frame count lost for mis-detections
                for j in remaining_ids:
                    tracked_detection_results[j, 15] += 1

                # Remove IDs that has no association for >= max_lost_frame
                max_lost_frame = 10
                tracked_detection_results = tracked_detection_results[(tracked_detection_results[:, 15] < max_lost_frame), :] 


        else:
            tracked_detection_results = new_detection_results        
        
        self.detection_history = tracked_detection_results
        return tracked_detection_results

