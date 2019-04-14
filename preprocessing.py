import sys
import os
import cv2
import numpy as np

class PreProcessing():
    def __init__(self):
        pass

    """
        Input: path to where the image is
        Return: matrix of the cropped image
    """
    def cropPieceFromImage(self,pathToImage):
        im = cv2.imread(os.path.abspath(pathToImage))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(im, (5, 5), 0)
        canny = cv2.Canny(blurred, 20, 40)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(canny, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        cnt = cnts[0]

        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        im = im[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
        im = cv2.GaussianBlur(im,(5,5),0)
        return im
   
   
   
   
#Checking where points in the image are
#print(leftmost, rightmost, topmost, bottommost)
# cv2.drawContours(im,cnts,0,(0,255,0),3)
# cv2.imwrite('res.jpg',im)
# cv2.circle(im, leftmost, 8, (0, 0, 255), -1)
# cv2.circle(im, rightmost, 8, (255, 255, 0), -1)
# cv2.circle(im, topmost, 8, (255, 0, 0), -1)
# cv2.circle(im, bottommost, 8, (0, 0, 0), -1)


# cv2.drawContours(im,cnts,0,(0,255,0),3)
# dilated = cv2.resize(im[topmost[1]-100:bottommost[1]+100, leftmost[0]-100:rightmost[0]+100],(524,524))
# dilated = cv2.resize(im[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]],(524,524))
