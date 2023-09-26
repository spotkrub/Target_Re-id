import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import OOI_gt


def gen_all_pixels(cam, num_x_px, num_y_px, n):
    num_x_px_per_n, num_y_px_per_n = int(num_x_px / n + 1), int(num_x_px / n + 1)

    OOI_world_coord_all = []
    for iA in range(num_x_px_per_n):
        for jA in range(num_y_px_per_n):
            OOI_world_coord_all.append([iA * n, jA * n])

    OOI_world_coord_all = np.array(OOI_world_coord_all, dtype=int)
    return OOI_world_coord_all

def draw_ooi_id(img, OOI_im_coords, OOI_ids, fontColor):
    img_cp = np.copy(img)
    
    font                   = cv.FONT_HERSHEY_SIMPLEX
    fontScale              = 2
    thickness              = 3
    lineType               = 2

    OOI_im_coords = np.array(OOI_im_coords, dtype='int')
    for i, OOI_coord in enumerate(OOI_im_coords):
        text_coord = OOI_coord.copy()
        text_coord[0] -= 18
        text_coord[1] -= 15
        
        cv.circle(img_cp, tuple(OOI_coord), 5, fontColor, -1)
        cv.putText( img_cp, 
                    str(int(OOI_ids[i])), 
                    text_coord, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                    bottomLeftOrigin=False
                    )
    return img_cp

def plot_original_id(imgA, imgB, results_ooiA, results_ooiB):
    fontColor = (0, 0, 255) # Red
    
    # Color OOI coordinates on image A
    imgA_drawn = draw_ooi_id(imgA, results_ooiA[:, 7:9], results_ooiA[:, 0], fontColor)

    # Color OOI coordinates on image B
    imgB_drawn = draw_ooi_id(imgB, results_ooiB[:, 7:9], results_ooiB[:, 0], fontColor)

    # Resize and save image
    imgA_resize = cv.resize(imgA_drawn, (0, 0), None, .50, .50)
    imgB_resize = cv.resize(imgB_drawn, (0, 0), None, .50, .50)
    combined_images = np.concatenate((imgA_resize, imgB_resize), axis=1)
    cv.imwrite('plots/original_ids.png', combined_images)

def plot_reid(imgA, imgB, results_ooiA, results_ooiB):
    fontColor_overlap = (0, 255, 0)
    fontColor_non_overlap = (255, 0, 255)

    # Color OOI coordinates on image A
    overlap_idA = results_ooiA[:, 13] >= 0
    non_overlap_idA = np.invert(overlap_idA)
    imgA_drawn = draw_ooi_id(imgA, results_ooiA[non_overlap_idA, 7:9], results_ooiA[non_overlap_idA, 13], fontColor_non_overlap)
    imgA_drawn = draw_ooi_id(imgA_drawn, results_ooiA[overlap_idA, 7:9], results_ooiA[overlap_idA, 13], fontColor_overlap)

    # Color OOI coordinates on image B
    overlap_idB = results_ooiB[:, 13] >= 0
    non_overlap_idB = np.invert(overlap_idB)
    imgB_drawn = draw_ooi_id(imgB, results_ooiB[non_overlap_idB, 7:9], results_ooiB[non_overlap_idB, 13], fontColor_non_overlap)
    imgB_drawn = draw_ooi_id(imgB_drawn, results_ooiB[overlap_idB, 7:9], results_ooiB[overlap_idB, 13], fontColor_overlap)
   
    # Resize and save image
    imgA_resize = cv.resize(imgA_drawn, (0, 0), None, .50, .50)
    imgB_resize = cv.resize(imgB_drawn, (0, 0), None, .50, .50)
    combined_images = np.concatenate((imgA_resize, imgB_resize), axis=1)
    cv.imwrite('plots/re_ids.png', combined_images)



""" World plot """
def plot_world(camA, camB, results_ooiA, results_ooiB, transformed_OOIs):
    # Plot settings
    ms = 70 # Marker size
    cA, cB, cgt = 'b', 'g', 'r' # Marker color for OOI in camA, OOI in camB, Ground truth OOI

    plt.figure()
    ax = plt.gca()
    plt.xlabel('[m]'), plt.ylabel('[m]'), plt.title('OOI in world coordinates')    
    ax.set_aspect('equal'), ax.grid()
    
    """ Plot all projected pixels """
    # # Resolution of the cameras
    # num_xA_px, num_yA_px = camA.info['res_x_px'], camA.info['res_y_px']
    # num_xB_px, num_yB_px = camB.info['res_x_px'], camB.info['res_y_px']    
    # n = 60   # nth pixel to next pixel
    # OOI_camA_pixels = gen_all_pixels(camA, num_xA_px, num_yA_px, n)
    # OOI_camB_pixels = gen_all_pixels(camA, num_xB_px, num_yB_px, n)
    # OOI_world_coordA_all = camA.im_to_world(OOI_camA_pixels)
    # OOI_world_coordB_all = camB.im_to_world(OOI_camB_pixels)
    # ax.scatter(OOI_world_coordA_all[:, 0], OOI_world_coordA_all[:, 1], marker='+', c=cA, label='camA all pixels') # Plot all pixels to world frame
    # ax.scatter(OOI_world_coordB_all[:, 0], OOI_world_coordB_all[:, 1], marker='x', c=cB, label='camB all pixels')
    
    """ Plot camera FOV """
    # Boundary pixels of cameras 
    # boundary_im_coordA_box = np.array([
    #                                     [0, 0],
    #                                     [num_xA_px, 0],
    #                                     [num_xA_px, num_yA_px],
    #                                     [0, num_yA_px],
    #                                     [0, 0], # Duplicate of first point to 'complete the box'
    #                                     ], dtype=int)

    # boundary_im_coordB_box = np.array([
    #                                     [0, 0],
    #                                     [num_xB_px, 0],
    #                                     [num_xB_px, num_yB_px],
    #                                     [0, num_yB_px],
    #                                     [0, 0]  # Duplicate of first point to 'complete the box'
    #                                     ], dtype=int)
    # boundary_im_coordA_box = camA.im_to_world(boundary_im_coordA_box)
    # boundary_im_coordB_box = camB.im_to_world(boundary_im_coordB_box)
    # ax.plot(boundary_im_coordA_box[:, 0], boundary_im_coordA_box[:, 1], c=cA, label='camA boundary') # Plot boundary pixels A to world frame
    # ax.plot(boundary_im_coordB_box[:, 0], boundary_im_coordB_box[:, 1], c=cB, label='camB boundary') # Plot boundary pixels B to world frame
    
    """Plot principal point"""
    # center0 = np.array([[num_xA_px / 2, num_yA_px / 2]], dtype=float)
    # center1 = np.array([[num_xB_px / 2, num_yB_px / 2]], dtype=float)
    # camA_principal = camA.im_to_world(center0)
    # camB_principal = camB.im_to_world(center1)
    # ax.scatter(camA_principal[:, 0], camA_principal[:, 1], marker='D', s=ms, c=cA, label='camA principal') # Plot camera A center to world frame
    # ax.scatter(camB_principal[:, 0], camB_principal[:, 1], marker='D', s=ms, c=cB, label='camB principal') # Plot camera B center to world frame
    
    """ Plot OOI """ 
    # # Ground truth
    # ax.scatter(OOI_gt.xyz[:, 0], OOI_gt.xyz[:, 1], marker='*', s=ms*4, c=cgt, label='Ground Truth (' + str(len(OOI_gt.xyz)) + ')') # Plot ground truth

    col_start = 9
    dim = np.shape(transformed_OOIs)[1]

    # Camera A
    OOI_world_coordA = results_ooiA[:, col_start:col_start+dim]
    ax.scatter(OOI_world_coordA[:, 0], OOI_world_coordA[:, 1], marker='^', s=ms, c='w', linewidths=1.2, edgecolors=cA, label='camA OOI (' + str(len(OOI_world_coordA)) + ')') # Plot camA detected OOI    
    for OOI_infoA in results_ooiA:
        plt.text(OOI_infoA[9] - 0.3, OOI_infoA[10] + 0.5, s=str(int(OOI_infoA[13])), c=cA) # Plot ID text
    
    # Camera B
    OOI_world_coordB = results_ooiB[:, col_start:col_start+dim]
    ax.scatter(OOI_world_coordB[:, 0], OOI_world_coordB[:, 1], marker='v', s=ms, c='w', linewidths=1.2, edgecolors=cB, label='camB OOI (' + str(len(OOI_world_coordB)) + ')')
    for OOI_infoB in results_ooiB:
        plt.text(OOI_infoB[9] - 0.3, OOI_infoB[10] - 1.3, s=str(int(OOI_infoB[13])), c=cB) # Plot ID
    
    # Transformed
    ax.scatter(transformed_OOIs[:, 0], transformed_OOIs[:, 1], marker='.', s=ms, c='k', linewidths=1.2, edgecolors='k', label='Xformed OOI (' + str(len(transformed_OOIs)) + ')') # Plot transformed detected OOI
    for OOI_transformed in transformed_OOIs:
        plt.text(OOI_transformed[0] + 0.3, OOI_transformed[1] - 0.4, s=str(int(OOI_transformed[2])), c='k') # Plot ID

    # # Specify plot range
    # min_x, max_x, min_y, max_y = -11, 12, -11, 12 
    # tick_steps = 1
    # plt.xlim([min_x, max_x]), plt.ylim([min_y, max_y])
    # plt.xticks(np.arange(min_x, max_x, tick_steps)), plt.yticks(np.arange(min_y, max_y, tick_steps))

    ax.legend(loc=(0.83, 0))
    plt.savefig('plots/world.png')
    plt.close()

def pc_dist(points1, points2):

    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    mean_dist = np.linalg.norm(centroid1 - centroid2)

    return mean_dist

""" Plot camA and camB world separately""" 
def plot_separate(camA, camB, results_ooiA, results_ooiB, transformed_OOIs):
    # Plot settings
    ms = 70 # Marker size
    cA, cB, cgt = 'b', 'g', 'r' # Marker color for OOI in camA, OOI in camB, Ground truth OOI

    plt.figure()

    # CamA
    plt.subplot(121)
    ax = plt.gca()
    plt.xlabel('[m]'), plt.ylabel('[m]'), plt.title('OOI A in world coordinates')    
    ax.set_aspect('equal'), ax.grid()
    ax.scatter(results_ooiA[:, 9], results_ooiA[:, 10], marker='^', s=ms, c='w', linewidths=1.2, edgecolors=cA)
    for OOI_infoA in results_ooiA:
        plt.text(OOI_infoA[9] - 0.3, OOI_infoA[10] + 0.5, s=str(int(OOI_infoA[13])), c=cA) # Plot ID text

    # CamB
    plt.subplot(122)
    ax = plt.gca()
    plt.xlabel('[m]'), plt.ylabel('[m]'), plt.title('OOI B in world coordinates')    
    ax.set_aspect('equal'), ax.grid()
    ax.scatter(results_ooiB[:, 9], results_ooiB[:, 10], marker='v', s=ms, c='w', linewidths=1.2, edgecolors=cB)
    for OOI_infoB in results_ooiB:        
        plt.text(OOI_infoB[9] - 0.3, OOI_infoB[10] - 1.3, s=str(int(OOI_infoB[13])), c=cB) # Plot ID

    # # Transformed
    # col_start = 9
    # dim = np.shape(transformed_OOIs)[1]

    # dist_to_A = pc_dist(transformed_OOIs, results_ooiA[:, col_start:col_start+dim])
    # dist_to_B = pc_dist(transformed_OOIs, results_ooiB[:, col_start:col_start+dim])
    

    # if dist_to_B < dist_to_A:
    #     ax.scatter(transformed_OOIs[:, 0], transformed_OOIs[:, 1], marker='.', s=ms, c='k', linewidths=1.2, edgecolors='k', label='Xformed OOI (' + str(len(transformed_OOIs)) + ')') # Plot transformed detected OOI
    #     for OOI_transformed in transformed_OOIs:
    #         plt.text(OOI_transformed[0] + 0.3, OOI_transformed[1] - 0.4, s=str(int(OOI_transformed[2])), c='k') # Plot ID
    # else:
    #     plt.subplot(121)
    #     ax = plt.gca()
    #     ax.scatter(transformed_OOIs[:, 0], transformed_OOIs[:, 1], marker='.', s=ms, c='k', linewidths=1.2, edgecolors='k', label='Xformed OOI (' + str(len(transformed_OOIs)) + ')') # Plot transformed detected OOI
    #     for OOI_transformed in transformed_OOIs:
    #         plt.text(OOI_transformed[0] + 0.3, OOI_transformed[1] - 0.4, s=str(int(OOI_transformed[2])), c='k') # Plot ID        

    plt.savefig('plots/world_separate.png')
    plt.close()



def plot_results(camA, camB, results_ooiA, results_ooiB, transformed_OOIs):
    plot_world(camA, camB, results_ooiA, results_ooiB, transformed_OOIs)
    plot_separate(camA, camB, results_ooiA, results_ooiB, transformed_OOIs)




 