import cv2
import os
from keras.preprocessing.image import img_to_array
from preprocessing import PreProcessing	
from neuralNetwork import NeuralNetwork

def createTrainImagesVectorAndLabels(pathToImageFolder):
    X_train = []
    Y_train = []
    highest = (0,0)
    for folder in os.listdir(pathToImageFolder):
        for image in os.listdir(os.path.abspath(os.path.join(pathToImageFolder,folder))):
            imagePath = os.path.abspath(os.path.join(os.path.join(pathToImageFolder,folder),image))
            im = cv2.imread(imagePath,0)
            # if im.shape > highest:
                # highest = cv2.imread(imagePath).shape
            im = cv2.resize(im,(400,400))
            X_train.append(im)
            Y_train.append(folder)
    return X_train, Y_train

if __name__ == "__main__":
    X_train, Y_train = createTrainImagesVectorAndLabels(os.path.abspath("real_Legos_images"))
    NN = NeuralNetwork()
    NN.t(X_train,Y_train)
    # preProcess = PreProcessing()
    # i = preProcess.cropPieceFromImage("photo5.jpg")
    # NN.teste("res.jpg")
    # preProcess.cropPieceFromImage("photo5.jpg")
    # cv2.imwrite('res.jpg',preProcess.cropPieceFromImage("photo5.jpg"))
