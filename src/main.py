#!/usr/bin/env python

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


import calibrage
import mise_correspondance
import carte_profondeur


def main():

    # paths
    calib_path = "images_calibrage"
    image_path = "images_figures"
    sauvegarde_path = "images_sauvegarde"

    # caracteristiques de calibrage
    nb_colonnes = 8
    nb_lignes = 5
    taille = (nb_colonnes, nb_lignes)

    # effectuer le calibrage
    matriceFondamentale, matriceEssentielle, matriceCamera_g, matriceCamera_d = calibrage.calibrage(calib_path, taille)

    # effectuer la validation
    calibrage.validation()

    # effectuer la mise en correspondance
    mise_correspondance.miseEnCorrespondance(image_path, matriceFondamentale)

    # effectuer la validation
    mise_correspondance.validation()

    # # evaluer les disparites
    # disp, dispbrute = carte_profondeur.disparite(i1, i2)

    # # effectuer le calcul de profondeur
    # carte_profondeur.calcul_profondeur(image_path + "/photoFlo1.jpg")

    # # effectuer la validation
    # carte_profondeur.validation()


if __name__ == "__main__":
    main()