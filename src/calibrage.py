#!/usr/bin/env python

import numpy as np
import cv2 as cv
import glob

#---------------------------------------------------------------------------#
# Code inspiré de https://learnopencv.com/camera-calibration-using-opencv/
#---------------------------------------------------------------------------#

# définir les criteres
criteres = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# vecteur 3D points de chaque image
objpoints = []
# vecteur 2D points de chaque image
imgpoints_g = []
imgpoints_d = []

# image size
imgSize = [1280, 960]


def calibrage(path_calib, taille):

    print("#---------------------------------------------------------------------------#")
    print("# ------------------------ Début du calibrage ------------------------------")
    print("#---------------------------------------------------------------------------#")

    images = glob.glob("./" + path_calib + "/*.jpg")

    print("# ------------------------ Recherche des coins ------------------------------")

    for fname in images:

        # charger l'image
        img = cv.imread(fname)

        # conversion en gris
        gris_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # séparer en 2
        demie = len(img[0])/2
        img_g = gris_img[:, :int(demie)]
        img_d = gris_img[:, int(demie):]

        # trouver les coins
        estTrouve_g, coins_g = cv.findChessboardCorners(img_g, taille, None)
        estTrouve_d, coins_d = cv.findChessboardCorners(img_d, taille, None)

        if estTrouve_g  == True and estTrouve_d == True:

            # world coordinates pour 3D points
            objp = np.zeros((1, taille[0] * taille[1], 3), np.float32)
            objp[0,:,:2] = np.mgrid[0:taille[0], 0:taille[1]].T.reshape(-1, 2)

            objpoints.append(objp)

            # peaufiner coordonnées coins
            sub_coins_g = cv.cornerSubPix(img_g, coins_g, (11,11),(-1,-1), criteres)
            sub_coins_d = cv.cornerSubPix(img_d, coins_d, (11,11),(-1,-1), criteres)

            imgpoints_g.append(sub_coins_g)
            imgpoints_d.append(sub_coins_d)

            # visualisation
            # img1 = cv.drawChessboardCorners(img, taille, sub_coins_g, True)
            # img2 = cv.drawChessboardCorners(img, taille, sub_coins_d, True)

            # cv.imshow('img_g',img1)
            # cv.imshow('img_d',img2)
            # cv.waitKey(0)


    cv.destroyAllWindows()

    print("# ------------------------ Calibration ------------------------------")

    # The actual calibration of left camera
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objpoints, imgpoints_g, imgSize, None, None)
    heightL, widthL = img_g.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    # the actual calibration of right camera
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objpoints, imgpoints_d, imgSize, None, None)
    heightR, widthR = img_d.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    # Stereo vision calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo = (cv.TERM_CRITERIA_EPS +
                       cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objpoints, imgpoints_g, imgpoints_d, newCameraMatrixL, distL, newCameraMatrixR, distR, gris_img.shape[::-1], criteria_stereo, flags)
    print("retval:\n", retStereo)
    print("\nRight camera intrinsic:\n", newCameraMatrixL)
    print("\nLeft camera intrinsic:\n", newCameraMatrixR)
    print("\nFundamental:\n", fundamentalMatrix)
    print("\nEssential:\n", essentialMatrix)
    print("\nTranslation:\n", trans)
    print("\nRotation:\n", rot)
    print("\nDistorsion left:\n", distL)
    print("\nDistorsion right:\n", distR)

    print("#---------------------------------------------------------------------------#")
    print("# ------------------------- Fin du calibrage --------------------------------")
    print("#---------------------------------------------------------------------------#")

    return fundamentalMatrix, newCameraMatrixL, newCameraMatrixR

def validation():
    a="h"
