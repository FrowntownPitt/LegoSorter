import cv2
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from preprocessing import PreProcessing    
from neuralNetwork import NeuralNetwork
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from random import randint
from keras.utils import plot_model


def moveFiles():
    path = os.path.abspath("photos")
    p = PreProcessing()
    # for i in os.listdir(path):
    # for j in os.listdir(os.path.abspath(os.path.join(path,i))):
    #     image_path = os.path.abspath(os.path.join(os.path.join(path,i),j))
    #     im = p.cropPieceFromConveyorBelt(image_path)
    #     cv2.imwrite(os.path.abspath(os.path.join(os.path.join("cropped_real_legos",i),j)),im)

def preProcessRenderedImages():
    #Preprocess it to get only the piece and overwrite the file with new output
    p = PreProcessing()
    path = os.path.abspath(os.path.join("rendered_legos"))
    for i in os.listdir(path):
        folder_files = os.path.abspath(os.path.join(path,i))
        for j in os.listdir(folder_files):
            img = os.path.abspath(os.path.join(folder_files,j))
            cv2.imwrite(img,p.cropPieceFromImage(img))

if __name__ == "__main__":
    batch_size = 36
    EPOCHS = 10
    path = "rendered_legos"
    evaluate_path = "cropped_real_legos"
    size1, size2 = 200, 200

    NN = NeuralNetwork()
    validationDataGenerator = ImageDataGenerator(rescale=1./255, rotation_range=90, vertical_flip=True,horizontal_flip=True,fill_mode = 'nearest')
    gen = ImageDataGenerator(rescale=1./255, rotation_range=90, vertical_flip = True, horizontal_flip=True,fill_mode = 'nearest')
    train_generator = gen.flow_from_directory(os.path.abspath(os.path.join(path)),
                                              target_size = (size1,size2), color_mode = "rgb", batch_size = batch_size, class_mode='categorical')
    validation_generator = validationDataGenerator.flow_from_directory(os.path.abspath(os.path.join(evaluate_path)),
                                                                       target_size = (size1,size2), color_mode = "rgb", batch_size = batch_size, class_mode='categorical')
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

    num_classes = len(os.listdir(os.path.abspath(os.path.join(path))))

    # model = NN.multitask_model(IMAGE_SIZE, num_classes, ["a", "b", "c"])
    # plot_model(model, to_file='model.png')
    # print("train steps:", train_generator.n//train_generator.batch_size)

    model = NN.inceptionV3Model((size1,size2,3),num_classes)
    plot_model(model, to_file='scratch.png')

    filepath="weights-miniVGG.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir = './logs', batch_size = batch_size)
    callbacks_list = [checkpoint,tensorboard]
    model.fit_generator(train_generator, validation_data = validation_generator, validation_steps = validation_generator.n//validation_generator.batch_size,
                    steps_per_epoch = STEP_SIZE_TRAIN, epochs = EPOCHS, callbacks = callbacks_list)
    
