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
	batch_size = 40
	path = "rendered_legos"
	evaluate_path = "cropped_real_legos"
	NN = NeuralNetwork()
	validationDataGenerator = ImageDataGenerator(rescale=1./255, rotation_range=90, vertical_flip=True,horizontal_flip=True,fill_mode = 'nearest')
	gen = ImageDataGenerator(rescale=1./255, rotation_range=90, vertical_flip = True, horizontal_flip=True,fill_mode = 'nearest')
	train_generator = gen.flow_from_directory(os.path.abspath(os.path.join(path)),
	target_size = (224,224), color_mode = "grayscale", batch_size = batch_size, class_mode='categorical')
	validation_generator = validationDataGenerator.flow_from_directory(os.path.abspath(os.path.join(evaluate_path)),
	target_size = (224,224), color_mode = "grayscale", batch_size = batch_size, class_mode='categorical')
	
	# k=0
	# for i in train_generator:
	#     k+=1
	#     if k==2:
	#         break
	STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
	num_classes = len(os.listdir(os.path.abspath(os.path.join(path))))
	VGG16 = NN.modelFromScratch((224,224,1),num_classes)
	filepath="weights.hdf5"
	checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
	callbacks_list = [checkpoint,reduce_lr]
	VGG16.fit_generator(train_generator, validation_data = validation_generator, validation_steps = validation_generator.n//validation_generator.batch_size,
					steps_per_epoch = STEP_SIZE_TRAIN, epochs = 50, callbacks = callbacks_list)
	# VGG16.load_weights('weights.hdf5')
	# preProcess = PreProcessing()
	# #     i = preProcess.cropPieceFromImage('photo6.jpeg')
	# im = cv2.imread("cropped_real_legos/2x2_L/2x2_L_06.jpg",0)
	# im = cv2.resize(im,(224,224))
	# test_img = img_to_array(im)
	# test_img = np.expand_dims(np.array(test_img), axis=0)
	# result = VGG16.predict_classes(test_img)
	# label_map = (train_generator.class_indices)
	# print(label_map,result)
