import foolbox
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 10000000000000000000      
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


# instantiate model
keras.backend.set_learning_phase(0)
model = load_model('resnet_foolbox_test.h5')
print("Model loaded.")

# dimensions of our images.
img_width, img_height = 224, 224

epochs = 100
batch_size = 16

# data_dir
data_dir = "C:\\Users\\Ben\\Desktop\\data\\subfolder-9\\"
processed = 0
for i in range(0, 10):
	print("Computing predictions for the {}th subset".format(i))
	preds = {}
	directory = data_dir + str(i) + "\\"
	for filename in os.listdir(directory):
		fname = directory + filename
		img = Image.open(fname)
		img = img.resize((img_width, img_height))
		img = np.asarray(img)
		img = img/255
		try:
			img = np.reshape(img, [1, img_width, img_height, 3])
			probs = model.predict(img)
			# class_label = np.argmin(probs, axis=1)
			preds[fname] = probs
		except ValueError:
			preds[fname] = -1
		processed += 1
		if processed%500 == 0:
			print("Processed {} images".format(processed))
	processed = 0
	print("Saving...")
	with open("artifacts/preds_"+str(i), 'wb') as f:
		pickle.dump(preds, f)
	print("Saved")


