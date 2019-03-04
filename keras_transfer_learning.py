from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000000000000      


# path to the model weights files.
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = "C:\\Users\\Ben\\Desktop\\new_data\\train"
validation_data_dir = "C:\\Users\\Ben\\Desktop\\new_data\\validation"
nb_train_samples = 8105+13044
nb_validation_samples = 243+553
epochs = 500
batch_size = 16

# build the VGG16 network
model = applications.ResNet50(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
print('Model loaded.')

# for layer in model.layers[:-10]:
#     layer.trainable = False

# add custom layers
x = model.output
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(1, activation="softmax")(x)


model_final = Model(input=model.input, output=predictions)


model_final.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("resnet.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')

# fine-tune the model
model_final.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks = [checkpoint, early])