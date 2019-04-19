import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from preprocessing import PreProcessing	
from neuralNetwork import NeuralNetwork
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from random import randint

def moveFiles():
    path = os.path.abspath("fotos/2x3")
    p = PreProcessing()
    # for i in os.listdir(path):
    #     if not os.path.isdir(os.path.abspath(os.path.join("cropped_real_legos",i))):
    #         os.mkdir(os.path.abspath(os.path.join("cropped_real_legos",i)))
        # for j in os.listdir(os.path.abspath(os.path.join("fotos",i))):
        # im = p.cropPieceFromConveyorBelt(os.path.abspath(os.path.join(path,i)))
        # cv2.imwrite("cropped_real_legos/2x3/"+i,im)
    #         cv2.imwrite(os.path.abspath(os.path.abspath(os.path.join("cropped_real_legos",j))),
    #             p.cropPieceFromImage(os.path.abspath(os.path.join(os.path.join("fotos",i),j))))
            # cv2.imwrite(os.path.abspath(os.path.abspath(os.path.join("cropped_real_legos",j)),p.cropPieceFromImage(os.path.abspath(os.path.join("fotos",i)))))
        # os.rename(os.path.abspath(os.path.join(p,vect[i])),os.path.abspath(os.path.join("2x2_L",vect[i])))

if __name__ == "__main__":
    # moveFiles()
    # batch_size = 60
    # path = "real_Legos_images/trainable_classes"
    # evaluate_path = "real_Legos_images/evaluation"
    # evaluate_path = "cropped_real_legos"
    # # X_train, Y_train = createTrainImagesVectorAndLabels(path)
    # NN = NeuralNetwork()
    p = PreProcessing()
    j = p.cropPieceFromConveyorBelt("photos/1x1/1x1_01.jpg")
    cv2.imwrite('res.jpg',j)
    gen = ImageDataGenerator(rotation_range=90, vertical_flip = True,
                    channel_shift_range = 60.0, width_shift_range=0.02, 
                    shear_range=0.02,height_shift_range=0.02, horizontal_flip=True, fill_mode='nearest')
    # train_generator = gen.flow_from_directory(os.path.abspath(os.path.join(path)),
    #                 save_to_dir='images', save_prefix='aug',save_format = 'png', target_size = (224,224), color_mode = "rgb", batch_size = batch_size, class_mode='categorical')
    # validation_generator = gen.flow_from_directory(os.path.abspath(os.path.join(evaluate_path)),
    #                     target_size = (224,224), color_mode = "rgb", batch_size = 60, class_mode='categorical',save_to_dir='images', save_prefix='aug',save_format = 'png')
    # k=0
    # for i in validation_generator:
    #     k+=1
    #     if k==2:
    #         break
    
    # STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    # num_classes = len(os.listdir(os.path.abspath(os.path.join(path))))
    # VGG16 = NN.modelFromScratch((224, 224, 1), num_classes)
    # filepath="weights-improvement-grayscale-balanced-dataset-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    # callbacks_list = [checkpoint]
    # VGG16.save_weights('weights.h5')
    # VGG16.fit_generator(train_generator, validation_data = validation_generator, validation_steps = validation_generator.n//validation_generator.batch_size,
                #    steps_per_epoch = STEP_SIZE_TRAIN, epochs = 50, callbacks = callbacks_list)
    # VGG16.load_weights('weights-improvement-38.hdf5')
    # preProcess = PreProcessing()
    # i = preProcess.cropPieceFromImage('photo6.jpeg')
    # i = cv2.resize(i,(224,224))
    # cv2.imwrite('res.jpg',i)
    # test_img = cv2.imread('res.jpg')
    # test_img = img_to_array(test_img)
    # test_img = np.expand_dims(np.array(test_img), axis=0)
    # result = VGG16.predict_classes(test_img)
    # label_map = (train_generator.class_indices)
    # print(label_map,result)
    
    # print(VGG16.predict_classes('res.jpg'))
