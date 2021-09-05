from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
import numpy as np

model = keras.Sequential()

model.add(Conv2D(6, (5, 5), input_shape=(32, 32, 1), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.SGD())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:, :, :, None] / 255. # 255. -> Float64 || image shape 28 x 28 x 1
x_test = x_test[:, :, :, None] / 255. 
x_train = tf.image.resize(x_train, (32, 32)) # 32 x 32 x 1
x_test = tf.image.resize(x_test, (32, 32))

# My PC is too weak Sadge.
model.fit(x_train, y_train, epochs=1)

z_test = model.predict(x_test)
accuracy = np.sum(z_test.argmax(axis=1) == y_test) / len(z_test) * 100
print(accuracy)