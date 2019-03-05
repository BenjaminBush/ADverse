import foolbox
import keras
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from PIL import Image

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = load_model('../resnet.h5')
print(kmodel.summary())
# kmodel = ResNet50(weights='imagenet')
# print(kmodel.summary())
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

# get source image and label
img_path = "C:\\Users\\Ben\\Desktop\\new_data\\validation\\ad\\170090.png"
image = Image.open(img_path)
image = image.resize((224, 224))
image = np.asarray(image, dtype=np.float32)
image = image[:, :, :3]
label = 0

# Get prediction for source image
pred_label = np.argmax(fmodel.predictions(image[:, :, ::-1]))
print(pred_label)

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image[:, :, ::-1], label)
# if the attack fails, adversarial will be None and a warning will be printed
# Get prediction for source image
adv_pred_label = np.argmax(fmodel.predictions(adversarial[:, :, ::-1]))



plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')
print(str(pred_label) + ", " + str(label))

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
plt.axis('off')
print(str(adv_pred_label) + ", " + str(label))

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial[:, :, ::-1] - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()