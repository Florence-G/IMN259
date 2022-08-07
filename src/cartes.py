import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def disparite(img_g, img_d):

    #---------------------------------------------------------------------------#
    # Code inspiré de https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
    #---------------------------------------------------------------------------#

    #Set disparity parameters
    #Note: disparity range is tuned according to specific parameters obtained through trial and error.
    win_size = 5
    min_disp = -1
    max_disp = 63 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16#Create Block matching object.
    stereo = cv.StereoSGBM_create(minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    uniquenessRatio = 5,
    speckleWindowSize = 5,
    speckleRange = 5,
    disp12MaxDiff = 1,
    P1 = 8*3*win_size**2,#8*3*win_size**2,
    P2 =32*3*win_size**2) #32*3*win_size**2)

    #Compute disparity map
    print ("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_g, img_d)

    # visualisation
    #plt.imshow(disparity_map,'gray')
    #plt.show()

    # sauvegarde
    cv.imwrite("./images_sauvegarde/disparity_map.jpg", disparity_map)

    return disparity_map

def validation():
    a="h"


