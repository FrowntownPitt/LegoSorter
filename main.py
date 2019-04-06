import sys
import os
import tensorflow as tf
import cv2
import numpy as np
from keras.applications import InceptionV3
import imutils
	
im = cv2.imread(os.path.abspath("photo3.jpg"))
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(img,100,200)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

accumEdged = np.zeros(im.shape[:2], dtype="uint8")
for chan in cv2.split(im):
	chan = cv2.medianBlur(chan, 11)
	edged = cv2.Canny(chan, 40, 180)
	accumEdged = cv2.bitwise_or(accumEdged, edged)
cv2.imwrite("res.jpg", accumEdged)
cnts = cv2.findContours(accumEdged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for i in cnts:
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.015 * peri, True)
    print(len(approx))
cv2.drawContours(im,cnts,-1,(0,255,0),3)
cv2.imwrite("res.jpg", im)