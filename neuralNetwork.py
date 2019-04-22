from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19, MobileNetV2
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam, SGD
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


    def multitask_model(self, input_shape, num_tasks, class_names):
        input_shape = Input(shape=input_shape, name="Input")

        x = Conv2D(8, kernel_size=(7, 7), activation='sigmoid', padding='same', strides=3)(input_shape)
        x = Conv2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=3)(x)

        x = Flatten()(x)
        x = Dense(64, activation='sigmoid', name='FeatureCollection')(x)

        tasks = []
        for t in range(num_tasks):
            task = Dense(32, activation='relu')(x)
            task = Dense(1, activation="linear", name=class_names[t])(task)
            tasks.append(task)

        # opt = SGD(lr = 0.01, decay = 0.01/20)
        model = Model(input_shape, tasks)
        model.summary()
        return model

    def add_new_task(self, model, old_task_names, new_task_name):
        input_shape = model.get_layer("Input")
        pre_task_layers = model.get_layer("FeatureCollection")
        old_tasks = []
        for t in old_task_names:
            old_tasks.append(model.get_layer(t).output)

        # freeze all old layers
        for layer in model.layers:
            layer.trainable = False

        new_task = Dense(32, activation='sigmoid')(pre_task_layers.output)
        new_task = Dense(1, activation="linear", name=new_task_name)(new_task)

        model = Model(input_shape.output, old_tasks + [new_task])
        model.summary()
        return model

    def unfreeze(self, model):
        for layer in model.layers:
            layer.trainable = True

        return model;

    def modelFromScratch(self,input_shape,num_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same',input_shape = input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

        model.add(Conv2D(128, (3, 3), padding="same", activation = "relu"))
        model.add(Conv2D(128, (3, 3), padding="same", activation = "relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        opt = Adam(lr=1e-4, decay=1e-4 / 12)
        # opt = SGD(lr = 0.01, decay = 0.01/20)
        model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["categorical_accuracy"])
        model.summary()
        return model

    def vgg16Model(self,image_shape,num_classes):
        model_VGG16 = VGG16(include_top = False, weights = None)
        model_input = Input(shape = image_shape, name = 'input_layer')
        output_VGG16_conv = model_VGG16(model_input)
        #Init of FC layers
        # x = GlobalAveragePooling2D()(output_VGG16_conv)
        x = Flatten(name='flatten')(output_VGG16_conv)
        x = Dense(512, activation = 'relu', name = 'fc1')(x)
        # x = Dense(128, activation = 'relu', name = 'fc2')(x)
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

    def inceptionV3Model(self, image_shape, num_classes):
        model_inceptionV3 = InceptionV3(include_top = False, weights = 'imagenet')
        x = model_inceptionV3.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation = 'relu')(x)
        predictions = Dense(num_classes, activation = 'softmax')(x)
        inception = Model(inputs = model_inceptionV3.input, output = predictions)
        for layer in inception.layers:
            layer.trainable = False
        inception.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics = ['categorical_accuracy'])
        return inception

    def mobileNetV2(self, image_shape, num_classes):
        base_model = MobileNetV2(include_top = False, weights = None)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation = 'relu', name = 'fc1')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss = 'categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['categorical_accuracy'])
        model.summary()
        return model
