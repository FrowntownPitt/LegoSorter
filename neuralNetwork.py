from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.datasets import mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
class NeuralNetwork():
    def __init__(self):
        pass

    def t(self,X_train,Y_train):
        trainX, testX, trainY, testY = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(400,400,3)))
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
        model.add(Dense(6))


        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
        nb_train_samples=400 
        nb_validation_samples = 100
        epochs = 10
        batch_size = 16
        train_generator = gen.flow_from_directory(
            os.path.abspath(os.path.join("real_Legos_images")), target_size=(400, 400), batch_size=16) 
        model.fit_generator( train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs)
        model.save_weights('first_try.h5')
        test_img = load_img('res.jpg',target_size=(400,400))
        test_img = img_to_array(test_img)
        result = model.predict(test_img)
        print(result)