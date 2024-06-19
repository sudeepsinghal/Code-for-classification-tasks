import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# shape and the dtype of images

print(train_images.shape)
print(train_images.dtype)

# plot the pictures of training dataset

classes = 10
f, ax = plt.subplots(nrows=1, ncols=classes, figsize=(10, 10))
for i in range(classes):
    sample = train_images[train_labels == i][0]
    ax[i].imshow(sample, cmap="gray")
    ax[i].set_title("label = {}".format([i]), fontsize=10)
plt.show()

# data preparation


# reshaping and converting datatype of training dataset
# data normalization( dividing the value of training dataset so that it is in the range of 0-1)
# normalization helps activation functions like sigmoid, GD to find the converging values faster

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images.astype("float32")/255.0
test_images = test_images.astype("float32")/255.0


# one-hot encode labels
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# NN

from tensorflow.keras.models import Sequential
from keras import layers

# detecting features
model = Sequential()
# for feature detection , we are using conv2D layer
model.add(layers.Conv2D(64, (3, 3), activation="leaky_relu", input_shape=(28, 28, 1)))
# decreases the size of the input by half(28x28) to (14x14)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# adding another conv2D layer
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="leaky_relu"))
# adding another max pooling layer to reduce size
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# last conv layer
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="leaky_relu"))

# flattening the layer so that it can be passed to the dense layers

model.add(layers.Flatten())
model.add(layers.Dense(units=784, activation="leaky_relu"))
model.add(layers.Dense(units=512, activation="leaky_relu"))
model.add(layers.Dense(units=256, activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

# training the model
batch_size = 128
epochs = 5
history = model.fit(train_images, train_labels, batch_size, epochs, verbose=1)

from keras import backend as k
k.clear_session()

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Loss : {} , Test Accuracy : {}'.format(test_loss, test_accuracy))

