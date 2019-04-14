import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from preprocessing import PreProcessing	
from neuralNetwork import NeuralNetwork
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
def createTrainImagesVectorAndLabels(pathToImageFolder):
    X_train = []
    Y_train = []
    highest = (0,0)
    for folder in os.listdir(pathToImageFolder):
        for image in os.listdir(os.path.abspath(os.path.join(pathToImageFolder,folder))):
            imagePath = os.path.abspath(os.path.join(os.path.join(pathToImageFolder,folder),image))
            im = cv2.imread(imagePath,0)
            X_train.append(im)
            Y_train.append(folder)
    return X_train, Y_train


if __name__ == "__main__":
    
    path = "real_Legos_images/trainable_classes"
    evaluate_path = "real_Legos_images/evaluation"
    # X_train, Y_train = createTrainImagesVectorAndLabels(path)
    NN = NeuralNetwork()
    gen = ImageDataGenerator(rotation_range=90, vertical_flip = True,
                    width_shift_range=0.02, shear_range=0.02,height_shift_range=0.02, horizontal_flip=True, fill_mode='nearest')
    train_generator = gen.flow_from_directory(os.path.abspath(os.path.join(path)),
                    target_size = (224,224), color_mode = "rgb", batch_size = 16, class_mode='categorical')

    validation_generator = gen.flow_from_directory(os.path.abspath(os.path.join(evaluate_path)),
                    target_size = (224,224), color_mode = "rgb", batch_size = 16, class_mode='categorical')
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    num_classes = len(os.listdir(os.path.abspath(os.path.join(path))))
    VGG16 = NN.vgg16Model((224, 224, 3), num_classes)
    #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
    VGG16.save_weights('weights.h5')
    VGG16.fit_generator(train_generator, validation_data = validation_generator, validation_steps = validation_generator.n//validation_generator.batch_size,
                    steps_per_epoch = STEP_SIZE_TRAIN, epochs = 50)
    # resNet.load_weights('weights.h5')
    # preProcess = PreProcessing()
    # i = preProcess.cropPieceFromImage("photo2.jpg")
    # i = cv2.resize(i,(224,224))
    # cv2.imwrite('res.jpg',i)
    # test_img = cv2.imread('res.jpg',0)
    # test_img = img_to_array(test_img)
    # test_img = np.expand_dims(np.array(test_img), axis=0)
    # result = resNet.predict(test_img)
    # y_classes = result.argmax(axis=-1)
    # label_map = (train_generator.class_indices)
    # print(y_classes,label_map,result)
    
    # print(resNet.predict_classes('res.jpg'))
