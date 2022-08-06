import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import calibrage
import mise_correspondance
import calcul_profondeur

def main():

    # effectuer le calibrage
    F, m = calibrage.calibrage()

    # effectuer la validation
    calibrage.validation()

    # effectuer la mise en correspondance
    i1, i2, p1, p2 = mise_correspondance.miseEnCorrespondance(F)

    # effectuer la validation
    mise_correspondance.validation()

    # evaluer les disparités
    disp, dispbrute = calcul_profondeur.disparite(i1, i2)

    # effectuer le calcul de profondeur
    calcul_profondeur.calcul_profondeur(disp, m, "result/depthmapNORMED.jpg")

    # effectuer la validation
    calcul_profondeur.validation()


if __name__ == "__main__":
    main()