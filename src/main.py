#!/usr/bin/env python

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


import calibrage
import mise_correspondance
import carte_profondeur

# paths
calib_path = "images_calibrage"
image_path = "../images_figures"
sauvegarde_path = "../images_sauvegarde"

# caracteristiques de calibrage
nb_colonnes = 8
nb_lignes = 5
taille = (nb_colonnes, nb_lignes)

def main():

    # effectuer le calibrage
    F, mL, mR = calibrage.calibrage(calib_path, taille)
    print(F)
    print(mL)
    print(mR)

    # # effectuer la validation
    # calibrage.validation()

    # # effectuer la mise en correspondance
    # i1, i2, p1, p2 = mise_correspondance.miseEnCorrespondance(F)

    # # effectuer la validation
    # mise_correspondance.validation()

    # # evaluer les disparites
    # disp, dispbrute = carte_profondeur.disparite(i1, i2)

    # # effectuer le calcul de profondeur
    # carte_profondeur.calcul_profondeur(disp, m, sauvegarde_path + "/mapProfondeur.jpg")

    # # effectuer la validation
    # carte_profondeur.validation()


if __name__ == "__main__":
    main()