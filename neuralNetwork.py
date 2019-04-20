from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

class NeuralNetwork():
    def __init__(self):
        pass

    def modelFromScratch(self,input_shape,num_classes):
        # trainX, testX, trainY, testY = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
        model = Sequential()
        model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid", input_shape=input_shape))

        # here we stack two CONV layers on top of each other where
        # each layerswill learn a total of 32 (3x3) filters
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
 
		# increase the number of filters again, this time to 128
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))
        opt = Adam(lr=1e-4, decay=1e-4 / 50)
        model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

        return model

    def vgg16Model(self,image_shape,num_classes):
        model_VGG16 = VGG16(include_top = False, weights = None, pooling = 'max')
        model_input = Input(shape = image_shape, name = 'input_layer')
        output_VGG16_conv = model_VGG16(model_input)
        #Init of FC layers
        # x = GlobalAveragePooling2D()(output_VGG16_conv)
        x = Flatten(name='flatten')(output_VGG16_conv)
        x = Dense(512, activation = 'relu', name = 'fc1')(x)
        x = Dense(128, activation = 'relu', name = 'fc2')(x)
        output_layer = Dense(num_classes,activation='softmax',name='output_layer')(x)
        vgg16 = Model(inputs = model_input, outputs = output_layer)
        vgg16.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
        vgg16.summary()
        return vgg16

    def resNet50Model(self, image_shape, num_classes):
        model_resNet50 = ResNet50(include_top = False, weights = None)
        model_input = Input(shape = image_shape, name = 'input_layer')
        output_resnet50_conv = model_resNet50(model_input)
        #Init of FC layers
        x = Flatten(name='flatten')(output_resnet50_conv)
        x = Dense(256, activation = 'relu', name = 'fc1')(x)
        #x = Dense(2048, activation = 'relu', name = 'fc2')(x)
        output_layer = Dense(num_classes,activation='softmax',name='output_layer')(x)
        resNet50 = Model(inputs = model_input, outputs = output_layer)
        resNet50.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
        resNet50.summary()
        return resNet50
