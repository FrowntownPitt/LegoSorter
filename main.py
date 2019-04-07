import sys
import os
import cv2
import numpy as np
from preprocessing import PreProcessing	



if __name__ == "__main__":
    preProcess = PreProcessing()
    cv2.imwrite('res.jpg',preProcess.cropPieceFromImage("photo5.jpg"))