from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from PIL import Image
import pickle
Image.MAX_IMAGE_PIXELS = 10000000000000000000      


# path to the model weights files.

train_data_dir = "C:\\Users\\Ben\\Desktop\\new_data\\train"
validation_data_dir = "C:\\Users\\Ben\\Desktop\\new_data\\validation"
nb_train_samples = 8105+13044
nb_validation_samples = 243+553

# dimensions of our images.
img_width, img_height = 224, 224

epochs = 100
batch_size = 16

# build the VGG16 network
model = applications.resnet50.ResNet50(weights=None)
print('Model loaded.')

# add custom layers
x = model.layers[-2].output
predictions = Dense(2, activation="softmax", kernel_initializer="random_uniform")(x)


model_final = Model(input=model.input, output=predictions)


model_final.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

print(model_final.summary())

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("resnet.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')

print("Train generator classes: {}".format(train_generator.class_indices))


# fine-tune the model
history = model_final.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks = [checkpoint, early])

with open('history', 'wb') as f:
    pickle.dump(history.history, f)