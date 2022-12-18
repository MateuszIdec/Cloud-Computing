import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
original_path = "data/original"
modified_path = "data/modified"

original_images = []
modified_images = []

# Iterate through the original images and load them into a list
for image_name in os.listdir(original_path):
  # Load the image
  image = keras.preprocessing.image.load_img(os.path.join(original_path, image_name))
  image = keras.preprocessing.image.img_to_array(image)

  # Resize the image using the `tf.image.resize` function
  image = tf.image.resize(image, [150, 150])

  original_images.append(image)

# Iterate through the modified images and load them into a list
for image_name in os.listdir(modified_path):
  # Load the image
  image = keras.preprocessing.image.load_img(os.path.join(modified_path, image_name))
  image = keras.preprocessing.image.img_to_array(image)

  # Resize the image using the `tf.image.resize` function
  image = tf.image.resize(image, [150, 150])

  modified_images.append(image)

# Create labels for the original and modified images
original_labels = np.zeros(len(original_images))
modified_labels = np.ones(len(modified_images))

# Combine the original and modified images and labels
images = np.concatenate([original_images, modified_images])
labels = np.concatenate([original_labels, modified_labels])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Normalize the pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create an ImageDataGenerator object to augment the training data
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use the ImageDataGenerator object to generate augmented training examples
x_train = train_datagen.flow(x_train, y_train, batch_size=32)
# Define the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

#num_batches = len(x_train) // 32
# Train the model
model.fit_generator(x_train, steps_per_epoch=50, epochs=100)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)


# Save the model
model.save('model.h5')

# Load the saved model
loaded_model = keras.models.load_model('model.h5')

# Define the paths to the two images you want to classify
image_path_1 = "data/modified/flickr_0500.png"
image_path_2 = "data/original/flickr_0004.png"

# Load and preprocess the images
image_1 = keras.preprocessing.image.load_img(image_path_1, target_size=(150, 150))
image_1 = keras.preprocessing.image.img_to_array(image_1)
image_1 = image_1 / 255.0
image_1 = np.expand_dims(image_1, axis=0)

image_2 = keras.preprocessing.image.load_img(image_path_2, target_size=(150, 150))
image_2 = keras.preprocessing.image.img_to_array(image_2)
image_2 = image_2 / 255.0
image_2 = np.expand_dims(image_2, axis=0)

# Use the model to classify the images
prediction_1 = loaded_model.predict(image_1)
prediction_2 = loaded_model.predict(image_2)

# Print the predictions
print("Prediction for image 1:", prediction_1[0][0])
print("Prediction for image 2:", prediction_2[0][0])