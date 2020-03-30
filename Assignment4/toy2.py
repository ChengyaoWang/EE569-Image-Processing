import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



img1 = cv.imread('../scrRaw/Husky_1.jpg')          # queryImage
img2 = cv.imread('../scrRaw/Husky_2.jpg')          # trainImage


img = cv.imread('../scrRaw/Husky_1.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('../Python_Keypoints.jpg', img)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create(0)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('../Python_Matches.jpg', img3)