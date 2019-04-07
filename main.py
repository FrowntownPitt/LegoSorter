import sys
import os
import tensorflow as tf
import cv2
import numpy as np
from keras.applications import InceptionV3
import imutils
	
im = cv2.imread(os.path.abspath("photo2.jpg"))
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 20, 40)
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(canny, kernel, iterations=2)
(contours, hierarchy) = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

cv2.drawContours(im,cnts,0,(0,255,0),3)
cv2.imwrite('res.jpg',im)