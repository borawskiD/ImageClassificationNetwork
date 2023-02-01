import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28*28)
X_test = X_test.reshape(10000,28*28)
X_train = X_train/255
X_test = X_test/255
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
print(y_test[0:10])
model = Sequential()
model.add(Dense(units=800, activation='relu', input_shape=(28*28,)))
model.add(Dense(units=640, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer=RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train,
                    y_train,
                    batch_size=1000,
                    epochs=10,
                    validation_data=(X_test, y_test))

plt.imshow(X_train[0].reshape(28,28), cmap="gray")
plt.show()
predicted = model.predict(X_train[0].reshape(1,-1))
print("softmax:")
print(predicted)
print(np.argmax(predicted,axis=1))