import cv2
import os
from keras.preprocessing.image import img_to_array
from preprocessing import PreProcessing	
from neuralNetwork import NeuralNetwork
from keras.preprocessing.image import ImageDataGenerator

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
            # im = cv2.resize(im,(224,400))
            X_train.append(im)
            Y_train.append(folder)
    return X_train, Y_train

if __name__ == "__main__":
    # X_train, Y_train = createTrainImagesVectorAndLabels(os.path.abspath("real_Legos_images"))
    NN = NeuralNetwork()
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
    train_generator = gen.flow_from_directory(os.path.abspath(os.path.join("real_Legos_images")), target_size = (224,224), color_mode = "rgb", batch_size = 32, class_mode='categorical')
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    resNet = NN.resNet50Model((224, 224, 3), 6)
    resNet.fit_generator(train_generator, steps_per_epoch = STEP_SIZE_TRAIN, epochs = 10)
    # NN.t()
    # preProcess = PreProcessing()
    # i = preProcess.cropPieceFromImage("rendered_LEGO-brick-images/train/3004 Brick 1x2/0001.png")
    # i = preProcess.cropPieceFromImage("photo5.jpg")
    # preProcess.cropPieceFromImage("photo5.jpg")
    # cv2.imwrite('res.jpg',i)
