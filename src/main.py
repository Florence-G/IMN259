#!/usr/bin/env python

import cv2 as cv
import glob

import calibrage
import mise_correspondance
import cartes
import reconstruction_3D


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
    matriceFondamentale, transVec, matrice_g = calibrage.calibrage(calib_path, taille)

    images = glob.glob("./" + image_path + "/*.jpg")

    for fname in images:

        # charger l'image
        img = cv.imread(fname)
        print(img.shape)

        # conversion en gris
        gris_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # séparer en 2
        demie = len(img[0])/2
        img_g = gris_img[:, :int(demie)]
        img_d = gris_img[:, int(demie):]

        # effectuer la mise en correspondance
        mise_correspondance.miseEnCorrespondance(img_g, img_d, matriceFondamentale)

        # evaluer les disparites
        carte_disparite = cartes.disparite(img_g, img_d)

        # effectuer la reconstruction 3D
        reconstruction_3D.map_3D(img_g, carte_disparite, matrice_g)



if __name__ == "__main__":
    main()