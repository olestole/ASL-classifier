import os
from zipfile import ZipFile
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import scipy
import util

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

old_training_data_path = "./data/asl_alphabet_train/"
old_test_data_path = "./data/asl_alphabet_test/"

training_data_path = "./data/training/"
test_data_path = "./data/validation/"

source_training_data_path = os.path.join(old_training_data_path, "asl_alphabet_train/")
source_test_data_path = os.path.join(old_test_data_path, "asl_alphabet_test/")

target_size = (200, 200, 3)
# move_images_to_validation(amount)


# Use imagegenetor to generate a batch of images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TODO: Add augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    training_data_path,
    target_size=target_size[:-1],
    batch_size=64,
    class_mode='categorical',
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    test_data_path,
    target_size=target_size[:-1],
    batch_size=64,
    class_mode='categorical',
)

num_categories = 29

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=target_size),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_categories, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())




EPOCHS = 20

# Steps_per_epoch = (batch_size / training_size) in the training set
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=5,
)


def plot_metric(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

plot_metric(history, 'accuracy')
plot_metric(history, 'loss')