from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

# visulisiong the trainging datasets
flag = 10
f,ax = plt.subplots(nrows = 1 , ncols = flag , figsize = (20,20))
for i in range(flag):
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample,cmap = 'Reds')
    ax[i].set_title('label = {}' . format(i), fontsize = 20)
plt.show()

# converting the 1 channel grayscale images to 3 channel rgb images
x_train = np.stack((x_train,)*3 , axis = -1)
x_test = np.stack((x_test,)*3 , axis = -1)
x_train = x_train.reshape(x_train.shape[0] ,28,28,3)
x_test = x_test.reshape(x_test.shape[0] ,28,28,3)


# normalizing the data
x_train = x_train/255
x_test = x_test/255

# one-hot encoding the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# using the pretraied ResNet50 model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(ResNet50(include_top = False , weights = 'imagenet' , pooling = 'max'))
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 256 , activation = 'relu'))
# model.add(Dropout(0.15))
model.add(Dense(units = 10 , activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy' , optimizer = 'RMSProp' , metrics = ['accuracy'])
# freezing first layer to prevent overfitting , decrease time because it has already learnt those features
model.layers[0].trainable = False
print(model.summary())

from keras import backend as k
keras.backend.clear_session()

epochs = 15
batch_size = 32
history = model.fit(x_test,y_test , batch_size = batch_size , epochs =epochs)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Loss : {} , Test Accuracy : {}'.format(test_loss, test_accuracy))