import keras
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adadelta, Adam
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np


def SigmoidOptimizerTest(model):
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X_train,
                     y_train,
                     batch_size=1000,
                     epochs=10,
                     validation_data=(X_test, y_test))


def AdadeltaOptimizerTest(model):
    model.compile(optimizer=adadelta,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X_train,
                     y_train,
                     batch_size=1000,
                     epochs=10,
                     validation_data=(X_test, y_test))


def adamOptimizerTest(model):
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X_train,
                     y_train,
                     batch_size=1000,
                     epochs=10,
                     validation_data=(X_test, y_test))


def activationsTest():
    activations = ['sigmoid', 'tanh', 'linear', 'relu', 'softmax']
    history_list = []
    for act in activations:
        model = Sequential()
        model.add(Dense(units=3100, activation=act, input_shape=(32 * 32 * 3,)))
        model.add(Dense(units=640, activation=act))
        model.add(Dense(units=100, activation='softmax'))

        model.compile(optimizer=RMSprop(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        hist = model.fit(X_train,
                         y_train,
                         batch_size=1000,
                         epochs=10,
                         validation_data=(X_test, y_test))

        plt.show()
        predicted = model.predict(X_train[0].reshape(1, -1))
        print("softmax:")
        print(predicted)
        print(np.argmax(predicted, axis=1))
        history_list.append(hist)
    for i, history in enumerate(history_list):
        plt.plot(history.history['val_accuracy'], label=activations[i])


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
print(X_train.shape)
X_train = X_train.reshape(50000, 32 * 32 * 3)
X_test = X_test.reshape(10000, 32 * 32 * 3)
print(X_train.shape)
print(X_train[0])
X_train = X_train / 255
X_test = X_test / 255
print(y_test[0:10])
y_train = keras.utils.to_categorical(y_train, num_classes=100)
y_test = keras.utils.to_categorical(y_test, num_classes=100)
print(y_test[0:10])

sgd = SGD(lr=0.01, momentum=0.9)
adadelta = Adadelta(lr=1.0, rho=0.95)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model = Sequential()
model.add(Dense(units=3100, activation='sigmoid', input_shape=(32 * 32 * 3,)))
model.add(Dense(units=640, activation='sigmoid'))
model.add(Dense(units=100, activation='softmax'))
SigmoidOptimizerTest(model)

plt.legend()
plt.show()
