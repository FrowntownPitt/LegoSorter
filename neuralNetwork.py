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
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
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

    def modelFromScratch(self):
        # trainX, testX, trainY, testY = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(200,200,1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())

        # Fully connected layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16))


        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
        nb_train_samples=400
        nb_validation_samples = 100
        epochs = 10
        batch_size = 16
        train_generator = gen.flow_from_directory(os.path.abspath(os.path.join("rendered_LEGO-brick-images/train")), target_size=(200, 200), color_mode = "grayscale", batch_size=16)
        model.fit_generator( train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs)
        model.save_weights('first_try.h5')
        
        model.load_weights('first_try.h5')
        
        test_img = cv2.imread('res.jpg',0)
        test_img = cv2.resize(test_img,(200,200))
        test_img = img_to_array(test_img)
        test_img = np.expand_dims(np.array(test_img), axis=0)
        result = model.predict(test_img)
        y_classes = result.argmax(axis=-1)
        label_map = (train_generator.class_indices)
        print(y_classes,label_map,result)

    def resNet50Model(self, image_shape, num_classes):
        model_resNet50 = ResNet50(include_top = False, weights = None)
        model_input = Input(shape = image_shape, name = 'input_layer')
        output_resnet50_conv = model_resNet50(model_input)
        x = Flatten(name='flatten')(output_resnet50_conv)
        x = Dense(4096, activation = 'relu', name = 'fc1')(x)
        x = Dense(4096, activation = 'relu', name = 'fc2')(x)
        x = Dense(num_classes, activation = 'softmax', name = 'predictions')(x)
        resNet50 = Model(inputs = model_input, outputs = x)
        resNet50.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return resNet50


        