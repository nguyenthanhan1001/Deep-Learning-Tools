from keras import models
from keras import layers

from keras.applications import densenet

def create_model(image_size):
	densenet_conv = densenet.DenseNet201(weights='imagenet',
                              include_top=True,
                              input_shape=(image_size, image_size, 3))
	model = models.Sequential()
	model.add(densenet_conv)
	model.summary()
	return model

def preprocess(img_batch):
	return densenet.preprocess_input(img_batch)
	pass
