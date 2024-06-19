import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Data path
root_dir = 'C:\python\projects\data\caltech-101\\101_ObjectCategories'

# Data augmentation with ImageDataGenerator
img_generator = ImageDataGenerator(
    rotation_range= 30,
    brightness_range=(0.5,1),
    # shear_range=0.2,
    # zoom_range=0.2,
    channel_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2 # Randomly shift color channels
)

# Define training, validation, and test sets using flow_from_directory
img_generator_flow_train = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=25,
    shuffle=True,
    subset="training"
)

img_generator_flow_valid = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset="validation"
)

img_generator_flow_test = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset='validation'
)

# Get sample images and labels for visualization
imgs, labels = next(iter(img_generator_flow_train))
for img, label in zip(imgs, labels):
    plt.imshow(img)
plt.show()  # Uncomment to display sample images

# Define a convolutional neural network (CNN) model from scratch
model = tf.keras.Sequential([
    # Convolutional layers with ReLU activation and max pooling
    keras.layers.Conv2D(256, (3, 3), activation='leaky_relu', input_shape=(224, 224, 3)),
    # keras.layers.Conv2D(256, (3, 3), activation='leaky_relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (2, 2), activation='leaky_relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='leaky_relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='leaky_relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # Flatten the output of the convolutional layers
    keras.layers.Flatten(),

    # Dense layers with ReLU activation and dropout for regularization
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(102, activation='softmax')  # 102 output units for 101 classes + background
])

# Model summary
model.summary()

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0015),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model on the training data with validation data for monitoring
model.fit(img_generator_flow_train,
          validation_data=img_generator_flow_valid,
          epochs=50,
          steps_per_epoch=30)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(img_generator_flow_test)
print("Test loss: {}, Test accuracy:  {}".format(test_loss, test_acc))

# clearing cache
from keras import backend as K
K.clear_session()