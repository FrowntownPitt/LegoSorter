import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from preprocessing import PreProcessing    
from neuralNetwork import NeuralNetwork
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from random import randint
import random
from keras.utils import plot_model
from keras.optimizers import Adam
from imutils import paths

def image_generator(imagePaths, batch_size):

    while True:
        batch_count = 0
        batch_input = []
        batch_output = []
        for imagePath in imagePaths:
            batch_count += 1
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = img_to_array(image)

            expectation = [[0]] * num_classes
            category = types[imagePath.split("/")[-2]]
            expectation[category] = [1]

            batch_input += [image]
            batch_output += [expectation]

            if batch_count == batch_size:
                yield np.array(batch_input), np.array(batch_output)
                batch_input = []
                batch_output = []

def image_list(imagePaths):
    data = []
    expectations = [[] for _ in range(len(types))]
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = img_to_array(image)
        data.append(image)

        expectation = [0] * len(types)
        category = types[imagePath.split("/")[-2]]
        expectation[category] = 1
        for e in range(len(expectation)):
            expectations[e].append(expectation[e])
        expectations.append(expectation)

    expectations = {class_names[t]: np.array(expectations[t]) for t in range(len(class_names))}

    return data, expectations


batch_size = 27
path = "rendered_legos"
evaluate_path = "cropped_real_legos"
NN = NeuralNetwork()

trainImagePaths = sorted(list(paths.list_images('rendered_legos')))# + list(paths.list_images('rendered_new')))
validImagePaths = sorted(list(paths.list_images('cropped_real_legos')))# + list(paths.list_images('real_new')))

testImagePaths = sorted(list(paths.list_images('real_Legos_images/miniEval')))
testNewImagePaths = sorted(list(paths.list_images('real_Legos_images/miniEval')) + list(paths.list_images('real_Legos_images/newMiniEval')))

newImagePaths = sorted(list(paths.list_images('rendered_new')))
newValidImagePaths = sorted(list(paths.list_images('real_new')))

EPOCHS = 25
IMAGE_SIZE = (200, 200, 1)
# STEP_SIZE_TRAIN = len(imagePaths) // batch_size

random.seed(4149)
random.shuffle(trainImagePaths)
random.shuffle(validImagePaths)
random.shuffle(newImagePaths)

types = list(os.listdir('rendered_legos'))# + list(os.listdir('rendered_new'))
types = {types[index]:index for index in range(len(types))}
num_classes = len(types)

losses = {t: "mean_squared_error" for t in types.keys()}
loss_weights ={t: 1.0 for t in types.keys()}

class_names = list(types.keys())
model = NN.multitask_model(IMAGE_SIZE, num_classes, class_names)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
plot_model(model, to_file='model.png')

filepath="weights-FRONet.hdf5"

train, new_model, test = True, True, False

if train:
    # generator = image_generator(trainImagePaths, batch_size)
    # valid_generator = image_generator(validImagePaths, batch_size)
    # print(hasattr(generator, 'next'), hasattr(generator, '__next__'))

    data, expectations = image_list(trainImagePaths)
    validData, validExpectations = image_list(validImagePaths)

    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    callbacks_list = [checkpoint,reduce_lr]
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(np.array(data), expectations, validation_data = (np.array(validData), validExpectations),
                    epochs = EPOCHS, callbacks = callbacks_list, batch_size=batch_size)

    print("Old model evaluation: ")
    model.load_weights("" + filepath)

    preProcess = PreProcessing()

    data, expectations = image_list(testImagePaths)

    ev = model.evaluate(x=np.array(data), y=expectations, verbose=1)

    for i in range(len(ev)):
        print("%s: %f" % (model.metrics_names[i], ev[i]))

if new_model:
    new_class_name = os.listdir("rendered_new")[0]
    types[new_class_name] = num_classes
    num_classes += 1

    model.load_weights(filepath)
    plot_model(model, to_file='new_model.png')

    model = NN.add_new_task(model, class_names, new_class_name)
    class_names = list(types.keys())

    new_data, new_expectations = image_list(newImagePaths)
    new_valid_data, new_valid_expectations = image_list(newValidImagePaths)

    checkpoint = ModelCheckpoint("new-" + filepath, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    callbacks_list = [checkpoint,reduce_lr]
    optimizer = Adam(lr=0.01, decay=0.00005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(np.array(new_data), new_expectations, validation_data = (np.array(new_valid_data), new_valid_expectations),
                    epochs = EPOCHS, callbacks = callbacks_list, batch_size=batch_size, verbose=0)

    print("New model evaluation: ")
    model.load_weights("new-" + filepath)

    preProcess = PreProcessing()

    data, expectations = image_list(testNewImagePaths)

    ev = model.evaluate(x=np.array(data), y=expectations, verbose=1)

    for i in range(len(ev)):
        print("%s: %f" % (model.metrics_names[i], ev[i]))

    model = NN.unfreeze(model)
    callbacks_list = [checkpoint,reduce_lr]
    optimizer = Adam(lr=0.0001, decay=0.00001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(np.array(new_data), new_expectations, validation_data = (np.array(new_valid_data), new_valid_expectations),
                    epochs = EPOCHS, callbacks = callbacks_list, batch_size=batch_size, verbose=0)

    print("New model evaluation: ")
    model.load_weights("new-" + filepath)

    preProcess = PreProcessing()

    data, expectations = image_list(testNewImagePaths)

    ev = model.evaluate(x=np.array(data), y=expectations, verbose=1)

    for i in range(len(ev)):
        print("%s: %f" % (model.metrics_names[i], ev[i]))

if test:
    print("Old model evaluation: ")
    model.load_weights("" + filepath)

    preProcess = PreProcessing()

    data, expectations = image_list(testImagePaths)

    ev = model.evaluate(x=np.array(data), y=expectations, verbose=1)

    for i in range(len(ev)):
        print("%s: %f" % (model.metrics_names[i], ev[i]))

    # for testImageName in testImagePaths:
    #     im = preProcess.cropPieceFromImage(testImageName)
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     # im = cv2.imread("",0)
    #     im = cv2.resize(im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    #     test_img = img_to_array(im)
    #     test_img = np.expand_dims(np.array(test_img), axis=0)
    #     result = model.predict(test_img, verbose=1)
    #     label_map = (class_names)
    #     print(label_map,result)