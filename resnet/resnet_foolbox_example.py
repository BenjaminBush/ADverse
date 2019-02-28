import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import foolbox
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.resnet50 import decode_predictions

print("Loading model")

keras.backend.set_learning_phase(0)
kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0,255), preprocessing=preprocessing)

print("loading imagenet example")

image, label = foolbox.utils.imagenet_example()
print(np.argmax(fmodel.predictions(image[:, :, ::-1])), label)

print("fgsm attack")

# apply attack on source image
attack  = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image[:,:,::-1], label)

print("finished attack")

print("showing results")

# show results
print(np.argmax(fmodel.predictions(adversarial)))
print(foolbox.utils.softmax(fmodel.predictions(adversarial))[781])
adversarial_rgb = adversarial[np.newaxis, :, :, ::-1]
preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))

print("Visualize attack")

plt.subplot(1, 3, 1)
plt.imshow(image/255)

plt.subplot(1, 3, 2)
plt.imshow(adversarial/255)

plt.subplot(1, 3, 3)
plt.imshow(adversarial - image)
plt.show()