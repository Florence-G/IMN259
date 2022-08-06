import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def calcul_profondeur(filename):

     # charger l'image
    img = cv.imread(filename)

    # conversion en gris
    gris_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # séparer en 2
    demie = len(img[0])/2
    img_g = gris_img[:, :int(demie)]
    img_d = gris_img[:, int(demie):]
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_g,img_d)
    plt.imshow(disparity,'gray')
    plt.show()


def disparite():
    a="h"

def validation():
    a="h"