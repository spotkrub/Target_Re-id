import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

from time import time

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import cv2 as cv


from sklearn.neighbors import NearestNeighbors


def points_in_FOV(OOI_results, boundary_path):
    
    col_start = 9
    col_end = 11
    OOI_world = OOI_results[:, col_start:col_end]
    
    # Check OOIs detected in camB if visible in camA
    visibility = np.zeros((len(OOI_results), 1))
    OOI_results = np.concatenate((OOI_results, visibility), axis=1)
    for i, point in enumerate(OOI_world):
        if boundary_path.contains_point(point[0:2]):
            OOI_results[i, -1] = 1

    return OOI_results

def FOV_boundary_path(cam):
    # Initialise boundary points of camera A as an numpy array
    corner_pixels_im = np.array([
                                [0,                     0],
                                [cam.info['res_x_px'],  0],                             
                                [cam.info['res_x_px'],  cam.info['res_y_px']],
                                [0,                     cam.info['res_y_px']]
                                ], dtype=int)

    # Convert corner pixels to world coordinates
    corner_pixels_world = cam.im_to_world(corner_pixels_im)

    # Convert corner pixels to boundary path
    boundary_path = mplPath.Path(corner_pixels_world[:, 0:2])

    return boundary_path

def check_FOV(camA, results_ooiA, camB, results_ooiB):
    """To Check FOV"""
    # # Get corner pixels of cam
    # boundary_pathA = FOV_boundary_path(camA)
    # boundary_pathB = FOV_boundary_path(camB)

    # # Identify OOIs within FOV of respective cameras
    # results_ooiA = points_in_FOV(results_ooiA, boundary_pathB)
    # results_ooiB = points_in_FOV(results_ooiB, boundary_pathA)

    return results_ooiA, results_ooiB

def plot_icp_convergence(values):
    label = "norm"    
    fig = plt.figure(99, figsize=(10, 4))
    ax = fig.add_subplot() 
    ax.plot(values, label=label)
    ax.legend()
    ax.grid(True)
    fig.savefig('plots/icp_convergence.png')
    plt.close()


# Find the nearest neighbors in the target point cloud
def nearest_neighbor_query(src, dst):
    tree = KDTree(dst)
    distances, indices = tree.query(src)
    return distances, indices

def nearest_neighbor_query_og(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()

# Define the Huber loss function
def Huber_mask(residuals, delta):
    mask = np.abs(residuals) < delta
    return 0.5 * (residuals ** 2) * mask + delta * (np.abs(residuals) - 0.5 * delta) * (1 - mask)

def Huber_loss(src, dst, distances, indices):
    huber_delta = 20

    # Compute the residuals and weights using the Huber loss function
    residuals = dst[indices] - src
    weights = Huber_mask(distances, huber_delta) / distances

    # Compute the weighted mean of the residuals
    mean_residuals = np.average(residuals, axis=0, weights=weights)

    # Compute the weighted covariance matrix
    m = src.shape[1]
    H = np.zeros((m, m))
    for j in range(src.shape[0]):
        H += weights[j] * np.outer(src[j] - mean_residuals, dst[indices[j]] - np.mean(dst, axis=0))

    return H, mean_residuals

def estimate_transformation_matrix(source, target, H, m):
    # Compute the rotation matrix using the OLS method
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    centroid_src = np.mean(source, axis=0)
    centroid_dst = np.mean(target, axis=0)

    t = centroid_dst - np.dot(R, centroid_src)

    # Update the transformation matrix
    T = np.eye(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

# Ref: https://github.com/ClayFlannigan/icp.git
def best_fit_transform(A, B):
   # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def mean_distance(points1, points2):
    dis_array = []
    for i in range(points1.shape[0]):
        dif = points1[i] - points2[i]
        dis = np.linalg.norm(dif)
        dis_array.append(dis)

    return np.mean(np.array(dis_array))


# Ref: https://github.com/ClayFlannigan/icp.git
def ICP_with_euc_loss(A, B):
    max_iterations = 1000
    tolerance = 0.00001
    init_pose=None

    # get number of dimensions
    m = A.shape[1]

    # print(A)
    # print(' ')
    # print(B)
    # print(' ')

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    mean_error_iters = []
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor_query_og(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        src = np.dot(T, src)

        # check error
        # mean_error = mean_distance(src[:m, :].T, dst[:m, indices].T)
        mean_error = np.mean(distances)

        mean_error_iters.append(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m, :].T)

    C = np.transpose(src[0:m, :])
    # print(C)
    # plot_icp_convergence(mean_error_iters)
    
    return C

# def ICP_with_euc_loss2(camA, OOI_A, camB, OOI_B):
#     max_iterations = 1000
#     tolerance = 0.00

#     m = OOI_A.shape[1]
#     dst = np.ones((m+1, OOI_B.shape[0]))
#     dst[:m,:] = np.copy(OOI_B.T)      

#     prev_error = 0
#     mean_error_iters = []
#     for i in range(max_iterations):
#         # make points homogeneous, copy them to maintain the originals
#         src = np.ones((m + 1, OOI_A.shape[0]))
#         src[:m, :] = np.copy(OOI_A.T)
        
#         # find the nearest neighbors between the current source and destination points
#         distances, indices = nearest_neighbor_query(src[:m, :].T, dst[:m, :].T)

#         # compute the transformation between the current source and nearest destination points
#         T,_,_ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

#         # update the current source
#         src = np.dot(T, src)

#         # check error
#         mean_error = np.mean(distances)
#         mean_error_iters.append(mean_error)
#         if np.abs(prev_error - mean_error) < tolerance:
#         # if mean_error < tolerance:
#         # if mean_error == prev_error:
#             break
#         prev_error = mean_error

#     # calculate final transformation
#     T,_,_ = best_fit_transform(OOI_A, src[:m,:].T)
#     return np.transpose(src[0:m, :])

# def ICP_with_huber_loss(source, target):
#     # Error Set 1: Human 2 / 7 (For Huber loss: Error with 3D points. Fine with 2D points)
#     # Error Set 2: Human 4 / 7 (For Huber loss: SVD Error in ICP -- Only 1 overlap)
#     # Error Set 4: Ball 1 / 5 (For Huber loss: Error with points. Fine with 2D points)
#     # Left to debug 3d and SVD error for Huber Loss
        
#     max_iterations = 1000
#     tolerance = 0.001
#     m = source.shape[1]

#     # Begin the ICP iteration loop
#     mean_error_iters = []
#     for i in range(max_iterations):
#         # Find the nearest neighbors in the target point cloud
#         distances, indices = nearest_neighbor_query(source, target)

#         # Remove outlier with Huber loss
#         H, mean_residuals = Huber_loss(source, target, distances, indices)
    
#         # Estimate transformation matrix
#         T, _, _, = estimate_transformation_matrix(source, target, H, m)

#         # Apply the current transformation to the source point cloud
#         source = np.dot(T[:m, :m], source.T).T

#         # Check for convergence
#         curr_norm = np.linalg.norm(mean_residuals)
#         mean_error_iters.append(curr_norm)
#         if i >= 1:
#             if curr_norm < tolerance and curr_norm - mean_error_iters[-2] <0:
#                 break

#     return source


def local_reid(camA, camB, results_ooiA, results_ooiB):  
    # Eliminate non-overlapping points to reid
    results_ooiA, results_ooiB = check_FOV(camA, results_ooiA, camB, results_ooiB)

    # Get OOIs in overlapping FOV
    ooiA_seen_by_camB = results_ooiA[results_ooiA[:, 12] == 1, :]
    ooiB_seen_by_camA = results_ooiB[results_ooiB[:, 12] == 1, :]
    num_ooiA_seen_by_camB = len(ooiA_seen_by_camB)
    num_ooiB_seen_by_camA = len(ooiB_seen_by_camA)

    col_start = 9
    col_end = 11 # 11 for feeding in 2D points; 2 for feeding in 3D points

    # Reid main
    if num_ooiA_seen_by_camB != 0 or num_ooiB_seen_by_camA != 0:
        
        # Determine the set with less OOIs
        if num_ooiA_seen_by_camB > num_ooiB_seen_by_camA:
            """ A HAS MORE OOI THAN B """
            """ Transforming B to A """
            to_transform = ooiB_seen_by_camA
            to_remain = ooiA_seen_by_camB

            transformed_ooi = ICP_with_euc_loss(to_transform[:, col_start:col_end], to_remain[:, col_start:col_end])

            remained_ooi = to_remain[:, col_start:col_end]
            distance_matrix = cdist(transformed_ooi, remained_ooi) # Compute the Euclidean distance matrix between the two sets of points
            _, matched_ooi_of_remained = linear_sum_assignment(distance_matrix) # Solve the one-to-one matching problem using the Hungarian algorithm
            
            add_id = num_ooiB_seen_by_camA
            transformed_ooi = np.concatenate((transformed_ooi, np.array([results_ooiB[:, 13]]).T), axis=1)

            if camA.id < camB.id:
                # Follow camA ids
                for i, idx in enumerate(matched_ooi_of_remained):
                    results_ooiB[i, 13] = idx
                    transformed_ooi[i, -1] = idx
            else:
                # Follow camB ids
                results_ooiA[:, 13] = -1
                for i, idx in enumerate(matched_ooi_of_remained):
                    results_ooiA[idx, 13] = i

                for row in results_ooiA:
                    if row[-1] == -1:
                        row[-1] = add_id
                        add_id +=1


        else:
            """ B HAS MORE OOI THAN A """
            """ Transforming A to B """
            to_transform = ooiA_seen_by_camB
            to_remain = ooiB_seen_by_camA

            transformed_ooi = ICP_with_euc_loss(to_transform[:, col_start:col_end], to_remain[:, col_start:col_end])
            remained_ooi = to_remain[:, col_start:col_end]
            distance_matrix = cdist(transformed_ooi, remained_ooi) # Compute the Euclidean distance matrix between the two sets of points
            _, matched_ooi_of_remained = linear_sum_assignment(distance_matrix) # Solve the one-to-one matching problem using the Hungarian algorithm

            add_id = num_ooiA_seen_by_camB
            transformed_ooi = np.concatenate((transformed_ooi, np.array([results_ooiA[:, 13]]).T), axis=1)

            if camA.id < camB.id:
                # Follow camA ids
                results_ooiB[:, 13] = -1
                for i, idx in enumerate(matched_ooi_of_remained):
                    results_ooiB[idx, 13] = i

                for row in results_ooiB:
                    if row[-1] == -1:
                        row[-1] = add_id
                        add_id +=1
            else:
                # Follow camB ids
                for i, idx in enumerate(matched_ooi_of_remained):
                    results_ooiA[i, 13] = idx
                    transformed_ooi[i, -1] = idx
 
    
    return results_ooiA, results_ooiB, transformed_ooi


def add_gaussian_noise(cam):
    # loc: Mean ('centre') of distribution.
    # scale: Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    # size: Output shape.

    print(  "ORIGINAL " +
            "-- ROLL: " + str(round(cam.roll, 3)) + "; PITCH: " + str(round(cam.pitch, 3)) + "; YAW: " + str(round(cam.yaw, 3)) +
            "; WORLD_X: " + str(round(cam.worldx, 3)) + "; WORLD_Y: " + str(round(cam.worldy, 3)) + "; WORLD_Z: " + str(round(cam.worldz, 3)))
 
    cam.roll    = cam.info['pose']['roll0']   + np.random.uniform(0, 5)
    cam.pitch   = cam.info['pose']['pitch0']  + np.random.uniform(0, 5)
    cam.yaw     = cam.info['pose']['yaw0']    + np.random.uniform(0, 5)
    cam.worldx  = cam.info['pose']['worldx0'] + np.random.uniform(0, 5)
    cam.worldy  = cam.info['pose']['worldy0'] + np.random.uniform(0, 5)
    cam.worldz  = cam.info['pose']['worldz0'] + np.random.uniform(0, 5)

    # cam.roll    += np.random.uniform(5, 10)
    # cam.pitch   += np.random.uniform(5, 10)
    # cam.yaw     += np.random.uniform(5, 10)
    # cam.worldx  += np.random.uniform(5, 10)
    # cam.worldy  += np.random.uniform(5, 10)
    # cam.worldz  += np.random.uniform(5, 10)


    print(  "W EROOR " +
            "-- ROLL: " + str(round(cam.roll, 3)) + "; PITCH: " + str(round(cam.pitch, 3)) + "; YAW: " + str(round(cam.yaw, 3)) +
            "; WORLD_X: " + str(round(cam.worldx, 3)) + "; WORLD_Y: " + str(round(cam.worldy, 3)) + "; WORLD_Z: " + str(round(cam.worldz, 3)))
    
    print(' ')


def remove_gaussian_noise(cam):
    # Remove noise for debugging purposes

    print(  "ORIGINAL " +
            "-- ROLL: " + str(round(cam.info['pose']['roll0'], 3)) + "; PITCH: " + str(round(cam.info['pose']['pitch0'], 3)) + "; YAW: " + str(round(cam.info['pose']['yaw0'], 3)) +
            "; WORLD_X: " + str(round(cam.info['pose']['worldx0'], 3)) + "; WORLD_Y: " + str(round(cam.info['pose']['worldy0'], 3)) + "; WORLD_Z: " + str(round(cam.info['pose']['worldz0'], 3)))
 
    cam.roll    = cam.info['pose']['roll0']
    cam.pitch   = cam.info['pose']['pitch0']
    cam.yaw     = cam.info['pose']['yaw0']
    cam.worldx  = cam.info['pose']['worldx0']
    cam.worldy  = cam.info['pose']['worldy0']
    cam.worldz  = cam.info['pose']['worldz0']

    print(  "W NOISE REMOVED " +
            "-- ROLL: " + str(round(cam.roll, 3)) + "; PITCH: " + str(round(cam.pitch, 3)) + "; YAW: " + str(round(cam.yaw, 3)) +
            "; WORLD_X: " + str(round(cam.worldx, 3)) + "; WORLD_Y: " + str(round(cam.worldy, 3)) + "; WORLD_Z: " + str(round(cam.worldz, 3)))
    
    print(' ')


