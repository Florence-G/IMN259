#!/usr/bin/env python

import numpy as np
import cv2 as cv
import glob


def calibrage(path_calib, taille):

    print("#---------------------------------------------------------------------------#")
    print("# ------------------------ Début du calibrage ------------------------------")
    print("#---------------------------------------------------------------------------#")

    #---------------------------------------------------------------------------#
    # Code inspiré de https://learnopencv.com/camera-calibration-using-opencv/
    #---------------------------------------------------------------------------#


    images = glob.glob("./" + path_calib + "/*.jpg")

    # définir les criteres
    criteres = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # vecteur 3D points de chaque image
    objpoints = []
    # vecteur 2D points de chaque image
    imgpoints_g = []
    imgpoints_d = []

     # image size
    imgSize = (1280,960)

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
            img1 = cv.drawChessboardCorners(img, taille, sub_coins_g, True)
            img2 = cv.drawChessboardCorners(img, taille, sub_coins_d, True)
            cv.imwrite("./images_sauvegarde/corners.jpg", img1)

            # cv.imshow('img_g',img1)
            # cv.imshow('img_d',img2)
            # cv.waitKey(0)


    cv.destroyAllWindows()

    print("# ------------------------ Calibrage ------------------------------ #")

    # Calibrage
    retval_g, matriceCamera_g, distorsionCoeff_g, rotVec_g, transVec_g = cv.calibrateCamera(objpoints, imgpoints_g, imgSize, None, None)
    retval_d, matriceCamera_d, distorsionCoeff_d, rotVec_d, transVec_d = cv.calibrateCamera(objpoints, imgpoints_d, imgSize, None, None)

    # paufiner les matrices
    height_g, width_g = img_g.shape
    height_d, width_d = img_d.shape

    matriceCamera_g, _ = cv.getOptimalNewCameraMatrix(matriceCamera_g, distorsionCoeff_g, (width_g, height_g), 1, (width_g, height_g))
    matriceCamera_d, _ = cv.getOptimalNewCameraMatrix(matriceCamera_d, distorsionCoeff_d, (width_d, height_d), 1, (width_d, height_d))

    # Stereo vision calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    # Fixer les paramètres intrinsèques
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retvalStereo, matriceCamera_g, distorsionCoeff_g, matriceCamera_d, distorsionCoeff_d, rotVec, transVec, matriceEssentielle, matriceFondamentale = cv.stereoCalibrate(
        objpoints, imgpoints_g, imgpoints_d, matriceCamera_g, distorsionCoeff_g, matriceCamera_d, distorsionCoeff_d, gris_img.shape[::-1], criteria_stereo, flags)

    print("Erreur de projection:\n", retvalStereo)
    print("\nMatrice camera gauche:\n", matriceCamera_g)
    print("\nMatrice camera droite:\n", matriceCamera_d)
    print("\nMatrice fondamentale:\n", matriceFondamentale)
    print("\nMatrice essentielle:\n", matriceEssentielle)
    print("\nVecteur de translation:\n", transVec)
    print("\nVecteur de rotation:\n", rotVec)
    print("\nDistorsion gauche:\n", distorsionCoeff_g)
    print("\nDistorsion droite:\n", distorsionCoeff_d)

    validation_rotation(rotVec)
    validation_matrice_triangulaire(matriceCamera_d)
    validation_matrice_triangulaire(matriceCamera_g)

    print("#---------------------------------------------------------------------------#")
    print("# ------------------------- Fin du calibrage --------------------------------")
    print("#---------------------------------------------------------------------------#")

    return matriceFondamentale, transVec, matriceCamera_g


def validation_rotation(matrice_rotation):

    # Matrice de rotation est une matrice orthonormale
    transposee = np.transpose(matrice_rotation)
    I_test = np.dot(transposee, matrice_rotation)
    I_vrai = [[1,0,0],[0,1,0],[0,0,1]]
    for i in range(3):
      for j in range(3):
         if (abs(I_test[i][j] - I_vrai[i][j]) > 0.001):
            print("Erreur de validation : la matrice de rotation n'est pas orthonormale")
            return
    return

def validation_matrice_triangulaire(matrice_cam):

	# Matrice intrinsquèque est une matrice triangulaire supérieure, alors vérifier si la matrice retourne est triangulaire supérieure
    estTriangulaire = np.allclose(matrice_cam, np.triu(matrice_cam)) # check if upper triangular
    if not estTriangulaire:
        print("Erreur de validation : au moins une des matrices intrinsèques n'est pas triangulaire suppérieure")
