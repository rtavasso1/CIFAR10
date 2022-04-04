import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dir = Path(r'CIFAR\cifar-10-batches-py')
batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

train_batches = [unpickle(data_dir / batch[i]) for i in range(5)]
test_batch = unpickle(data_dir / 'test_batch')

# Stacks of 3x3 Convs w/ batch norm, dropout, and max pooling layers, 77% accuracy after 20 epochs
model = Sequential()
model.add(Rescaling(1./255))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

X = np.array([i[b'data'] for i in train_batches], dtype='float').reshape(50000,32,32,3)
y = np.array([i[b'labels'] for i in train_batches], dtype='float').reshape(50000,1)
X_val = np.array(test_batch[b'data'], dtype='float').reshape(10000,32,32,3)
y_val = np.array(test_batch[b'labels'], dtype='float').reshape(10000,1)
y, y_val = OneHotEncoder(sparse=False).fit_transform(y), OneHotEncoder(sparse=False).fit_transform(y_val)

model.fit(X, y, batch_size=32, epochs=20, validation_data=(X_val, y_val))