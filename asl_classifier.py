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

zip_path = "./archive.zip"
data_path = "./data/"

with ZipFile(zip_path, 'r') as zipObj:
    zipObj.extractall(data_path)
    print(f"Extracted to {data_path}")


# The extracted data is uneccesary nested - move the inner folder to the root of training/test folders respectively
old_training_data_path = "./data/asl_alphabet_train/"
old_test_data_path = "./data/asl_alphabet_test/"

training_data_path = "./data/training/"
test_data_path = "./data/validation/"

source_training_data_path = os.path.join(old_training_data_path, "asl_alphabet_train/")
source_test_data_path = os.path.join(old_test_data_path, "asl_alphabet_test/")

shutil.copytree(source_training_data_path, training_data_path)
shutil.copytree(source_test_data_path, test_data_path)

#shutil.rmtree(old_training_data_path)
#shutil.rmtree(old_test_data_path)

# Find all the different categories:
categories_list = []
for root, dirs, files in os.walk(training_data_path, topdown=False):
    if root.split("/")[-1] == "":
        continue
    categories_list.append(root.split("/")[-1])

categories_list.sort()
categories_dict = {key: value for (key, value) in enumerate(categories_list)}

num_categories = len(categories_dict)
print(f"Total number of categories: {num_categories}")

restructure_validation = True

# Need to put all the validation data into a single folder for the data generator to work
if restructure_validation:
    for category in categories_list:
        # Create the empty folder
        source_path = os.path.join(test_data_path, category)
        os.mkdir(source_path)

    # Move the file to corresponding folder
    for filename in os.listdir(test_data_path):
        file_path = os.path.join(test_data_path, filename)
        file_category = filename.split("_")[0]
        destination_path = os.path.join(test_data_path, file_category)
        shutil.move(file_path, destination_path)


# Get a random image from the training set and test set
def get_random_image(from_validation=False, seed=None):
    if (seed):
        random.seed(seed)
    random_category = random.choice(categories_list)
    random_category_path = os.path.join(test_data_path if from_validation else training_data_path, random_category)
    random_image_path = random.choice(os.listdir(random_category_path))
    image = mpimg.imread(os.path.join(random_category_path, random_image_path))
    return (image, random_category)

image, image_category = get_random_image()
target_size = image.shape

util.show_image(image, title=f"Random image from '{image_category}'")


random_images = np.asarray([get_random_image(from_validation=True, seed=i*42) for i in range(10)])
images, titles = random_images[:,0], random_images[:,1]

util.show_images(images, image_titles=titles)


# Randomly split some of the training data into validation data
def sample_images_from_category(category, amount, seed=None, from_validation=False):
    if (seed):
        random.seed(seed)
    category_path = os.path.join(test_data_path if from_validation else training_data_path, category)
    image_path_samples = random.sample(os.listdir(category_path), amount)
    return image_path_samples

# 10% of the training data will be used for validation
split = 0.1
amount = int(3000 * split) # TODO: Change this to be the len of some folder

def move_images_to_validation(amount):
    for category in categories_list:
        image_path_samples = sample_images_from_category(category, amount, seed=42)
        for image_path in image_path_samples:
            source_path = os.path.join(training_data_path, category, image_path)
            destination_path = os.path.join(test_data_path, category)
            shutil.move(source_path, destination_path)

# move_images_to_validation(amount)


# Use imagegenetor to generate a batch of images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TODO: Add augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    training_data_path,
    target_size=target_size[:-1],
    batch_size=100,
    class_mode='categorical',
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    test_data_path,
    target_size=target_size[:-1],
    batch_size=100,
    class_mode='categorical',
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=target_size),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_categories, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Checkpoint callback
checkpoint_path = "./data/training_checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)




EPOCHS = 20

# Steps_per_epoch = (batch_size / training_size) in the training set
history = model.fit(
    train_generator,
    steps_per_epoch=783,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=87,
    callbacks=[cp_callback]
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