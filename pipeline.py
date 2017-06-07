import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from moviepy.editor import *
from IPython.display import HTML
import cv2
from advance_lane_functions import *


src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
# window settings
window_width = 50
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching
set_prev = 0
def process_frame(image):
    #undistorting
    global left_fit_prev
    global right_fit_prev
    global set_prev
    undistorted_image = undistort(image)
    warped = warp_binarize_pipeline(image)
    undist = undistorted_image
    window_centroids,leftx_base,rightx_base = find_window_centroids(warped, window_width, window_height, margin)

    good_left_x = []
    good_left_y = []
    good_right_x = []
    good_right_y = []
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            l_mask2 = cv2.bitwise_and(warped,warped, mask=l_mask)
            good_l = np.argwhere(l_mask2>0)
            good_l_rev = []
            for k in good_l:
                good_left_x.append(k[0])
                good_left_y.append(k[1])
                good_l_rev.append([k[1],k[0]])
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            r_mask2 = cv2.bitwise_and(warped,warped, mask=r_mask)
            good_r = np.argwhere(r_mask2>0)
            good_r_rev = []
            for k in good_r:
                good_right_x.append(k[0])
                good_right_y.append(k[1])
                good_r_rev.append([k[1],k[0]])
            # Add graphic points from window mask here to total pixels found
            l_points[ ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    if len(good_left_x)>0 and len(good_right_x)>0:
        left_fit = np.polyfit(good_left_x, good_left_y, 2)
        right_fit = np.polyfit(good_right_x, good_right_y, 2)

        if set_prev == 0:
            set_prev = 1
            right_fit_prev = right_fit
            left_fit_prev  = left_fit
        ## Check error between current coefficient and on from previous frame
        err_p_R = np.sum((right_fit[0]-right_fit_prev[0])**2) #/np.sum(right_fit_prev[0]**2)
        err_p_R = np.sqrt(err_p_R)
        if err_p_R>.0005:
            right_fit = right_fit_prev
            #col_R = col_R_prev
        else:
            right_fit = .05*right_fit+.95*right_fit_prev

        right_fit_prev = right_fit
        left_fit_prev = left_fit
        ## Check error between current coefficient and on from previous frame
        err_p_L = np.sum((left_fit[0]-left_fit_prev[0])**2) #/np.sum(right_fit_prev[0]**2)
        err_p_L = np.sqrt(err_p_L)
        if err_p_L>.0005:
            left_fit =  left_fit_prev
            #col_L = col_L_prev
        else:
            left_fit =  .05* left_fit+.95* left_fit_prev

        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        persp_transform_image = warped
        warp_zero = np.zeros_like(persp_transform_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        #Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (persp_transform_image.shape[1], persp_transform_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        curvature_string = curvature_radius(persp_transform_image, left_fitx, right_fitx,ploty,good_left_x,good_right_x,good_left_y,good_right_y)
        location_string = pos_from_center(persp_transform_image, leftx_base, rightx_base)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(result,curvature_string,(400,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,location_string,(400,100), font, 1,(255,255,255),2,cv2.LINE_AA)

        return result
    else:
        return undistorted_image

def process_video(input_path, output_path):
    input_file = VideoFileClip(input_path)
    output_clip = input_file.fl_image(process_frame)
    output_clip.write_videofile(output_path, audio=False)

process_video('project_video.mp4', 'test_video_try_12.mp4')
