from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000000000000      


# path to the model weights files.
# dimensions of our images.
img_width, img_height = 224, 224
batch_size=16

train_data_dir = "C:\\Users\\Ben\\Desktop\\new_data\\train"
validation_data_dir = "C:\\Users\\Ben\\Desktop\\new_data\\validation"

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

label_map = (validation_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
print(label_map)

kmodel = load_model("resnet.h5")
eval_ = kmodel.evaluate_generator(validation_generator)
print(eval_)