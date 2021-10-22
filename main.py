import numpy as np
import pandas as pd
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from custom_metric import custom_metric_function

x_data = pd.read_csv("../train_X.csv", index_col=0, sep=',')
y_data = pd.read_csv("../train_y.csv", index_col=0, sep=',')

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_data, y_data)

x_valid = pd.read_csv("../train_X.csv", index_col=0, sep=',')
y_valid = pd.read_csv("../train_y.csv", index_col=0, sep=',')

x_test = pd.read_csv("../test_X.csv", index_col=0, sep=',')

model = Sequential()

model.add(Dense(100, activation="relu", input_dim = x_train.shape[1]))
model.add(Dense(100, activation="relu"))
model.add(Dense(248, activation="sigmoid"))
model.compile(loss=custom_metric_function, optimizer="adam", metrics="mae")

model.summary()

batch_size = 128
epochs = 12

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])