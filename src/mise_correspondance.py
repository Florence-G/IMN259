#!/usr/bin/env python

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def miseEnCorrespondance(img_g, img_d, F):

    print("#---------------------------------------------------------------------------#")
    print("# ----------------- Début de la mise en correspondance ----------------------")
    print("#---------------------------------------------------------------------------#")

    print("# ------------------------------- Ransac ------------------------------------")

    #---------------------------------------------------------------------------#
    # Code tiré de https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #---------------------------------------------------------------------------#

    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_g,None)
    kp2, des2 = sift.detectAndCompute(img_d,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img_g.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img_d,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv.drawMatches(img_g,kp1,img2,kp2,good,None,**draw_params)

    # visualisation
    #plt.imshow(img3, 'gray'),plt.show()

    # sauvegarde
    cv.imwrite("./images_sauvegarde/ransac.jpg", img3)


    print("# -------------- Brute-Force Matching with ORB Descriptors ------------------")

    #---------------------------------------------------------------------------#
    # Code tiré de https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #---------------------------------------------------------------------------#

    # Brute-Force Matching with ORB Descriptors

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_g,None)
    kp2, des2 = sift.detectAndCompute(img_d,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img_g,kp1,img_d,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # visualisation
    #plt.imshow(img3),plt.show()

    # sauvegarde
    cv.imwrite("./images_sauvegarde/brute-force.jpg", img3)


    print("# -------------------------- FLANN based Matcher ----------------------------")

    #---------------------------------------------------------------------------#
    # Code tiré de https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #---------------------------------------------------------------------------#

    # FLANN based Matcher

    # detecteur SIFT
    sift = cv.SIFT_create()

    #
    kp1, des1 = sift.detectAndCompute(img_g,None)
    kp2, des2 = sift.detectAndCompute(img_d,None)

    # parametres FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img_g,kp1,img_d,kp2,matches,None,**draw_params)

    # visualisation
    #plt.imshow(img3,),plt.show()

    # sauvegarde
    cv.imwrite("./images_sauvegarde/Flann.jpg", img3)

    # compute epipole
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img_g,img_d,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img_d,img_g,lines2,pts2,pts1)

    # visualisation
    #plt.subplot(121),plt.imshow(img5)
    #plt.subplot(122),plt.imshow(img3)
    #plt.show()

    # sauvegarde
    cv.imwrite("./images_sauvegarde/epipoles.jpg", img3)


    print("#---------------------------------------------------------------------------#")
    print("# ----------------- Fin de la mise en correspondance ----------------------")
    print("#---------------------------------------------------------------------------#")

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2







