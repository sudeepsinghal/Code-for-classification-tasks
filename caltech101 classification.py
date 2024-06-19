import os
import tensorflow as tf
# from tensorflow import keras
# from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
# os.environ['KAGGLE_CONFIG_DIR'] = '/content'

root_dir = 'C:\python\projects\data\caltech-101\\101_ObjectCategories'

img_generator =tf.keras.preprocessing.image.ImageDataGenerator(
    # rotation_range=90,
                                                                brightness_range=(0.5,1),
                                                                # shear_range=0.2,
                                                                # zoom_range=0.2,
                                                                channel_shift_range=0.2,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                rescale=1/255,
                                                                validation_split=0.2)
img_generator_flow_train = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset="training")

img_generator_flow_valid = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset="validation")
img_generator_flow_test = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset='validation')

imgs, labels = next(iter(img_generator_flow_train))
for img, label in zip(imgs, labels):
    plt.imshow(img)
    # plt.show()

from keras import layers

base_model = tf.keras.applications.InceptionV3(include_top=False,
                                               weights = "imagenet",
                                               classifier_activation = 'softmax')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(256, activation="relu"),
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(102, activation="softmax")
])

model.summary()

model.compile(optimizer="RMSProp",
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])



model.fit(img_generator_flow_train, validation_data=img_generator_flow_valid, steps_per_epoch=50, epochs=50)
test_acc , test_loss = model.evaluate(img_generator_flow_test)
print("Test loss: {}, Test accuracy:  {}".format(test_acc, test_loss))


# clearing cache
from keras import backend as K
K.clear_session()