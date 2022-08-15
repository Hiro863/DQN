from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from src.parameters import *


def create_cnn():
    model = Sequential()

    model.add(Conv2D(
                filters=32,
                input_shape=(INPUT_SIZE, INPUT_SIZE, CHANNEL_NUMBER),
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    ))

    model.add(Flatten())

    model.add(Dense(units=27))

    return model

def creat_fc():

    model_ = Sequential()

    model_.add(Dense(units=1024))

    model_.add(Dense(units=256))

    model_.add(Dense(units=5))

    opt = Adam(lr=L_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model_.compile(loss='mse', optimizer=opt)

    return model_


