import foolbox
import keras
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

Image.MAX_IMAGE_PIXELS = 10000000000000000000      


# instantiate model
keras.backend.set_learning_phase(0)
kmodel = load_model('../resnet_foolbox_test.h5')
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255))

# get source image and label
data_dir = "C:\\Users\\Ben\\Desktop\\data\\subfolder-9\\"
num_evasions = 0
num_files = 0
processed = 0
for i in range(0, 1):
	directory = data_dir + str(i) + "\\"
	f = open('../artifacts/preds_'+str(i), 'rb')
	preds = pickle.load(f)
	adversarial_images = {}
	for filename in os.listdir(directory):
		num_files += 1
		fname = directory + filename
		img = Image.open(fname)
		img = img.resize((224, 224))
		image = np.asarray(img, dtype=np.float32)
		try:
			image = image[:, :, :3]
		except:
			continue
		label = 0

		# Get prediction for source image
		pred_label = np.argmax(fmodel.predictions(image[:, :, ::-1]))
		print(pred_label)

		# apply attack on source image
		# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
		attack = foolbox.attacks.FGSM(fmodel)
		adversarial = attack(image[:, :, ::-1], label)
		processed += 1
		if adversarial is None:
			continue
		else:
			difference = adversarial[:, :, ::-1] - image
			if np.sum(difference) == 0:
				continue

			num_evasions += 1
			adversarial_images[fname] = adversarial

			# # if the attack fails, adversarial will be None and a warning will be printed
			# # Get prediction for source image
			adv_pred_label = np.argmax(fmodel.predictions(adversarial[:, :, ::-1]))



			# plt.figure()

			# plt.subplot(1, 3, 1)
			# plt.title('Original')
			# plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
			# plt.axis('off')
			# print(str(pred_label) + ", " + str(label))

			# plt.subplot(1, 3, 2)
			# plt.title('Adversarial')
			# plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
			# plt.axis('off')
			# print(str(adv_pred_label) + ", " + str(label))

			# plt.subplot(1, 3, 3)
			# plt.title('Difference')
			# difference = adversarial[:, :, ::-1] - image
			# plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
			# plt.axis('off')

			# plt.show()
		if processed % 100 == 0:
			print("Processed {}".format(processed))
			print("Evaded {}".format(num_evasions))

		# if processed > 500:
		# 	processed = 0
		# 	break


	print("Saving...")
	with open("../artifacts/adversarial_images_"+str(i), 'wb') as f:
		pickle.dump(adversarial_images, f)
	with open("../artifacts/stats_"+str(i), "w+") as f2:
		f2.write("Number of files: {}\n".format(num_files))
		f2.write("Number of evasions: {}\n".format(num_evasions))
	print("Saved")
	print(num_evasions)
