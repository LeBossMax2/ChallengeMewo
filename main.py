import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from custom_metric import custom_metric_function

x_data = pd.read_csv("../train_X.csv", index_col=0, sep=',')
y_data = pd.read_csv("../train_y.csv", index_col=0, sep=',')

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_data, y_data)

model = Sequential()

custom_loss = tf.keras.losses.BinaryCrossentropy()
custom_opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.add(Dense(300, activation="relu", input_dim = x_train.shape[1]))
model.add(Dense(280, activation="relu"))
model.add(Dense(248, activation="sigmoid"))
model.compile(loss=custom_loss, optimizer=custom_opt, metrics="mae")

model.summary()

batch_size = 10
epochs = 40

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid))

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

x_test = pd.read_csv("../test_X.csv", index_col=0, sep=',')

nb_split = 100

index_split = np.array_split(x_test.index, nb_split)

with open("../pred_y.csv", "w") as file:
    file.write("ChallengeID,")
    np.savetxt(file, y_data.columns.to_numpy().reshape(1, len(y_data.columns)), fmt='%s', delimiter=",")

    for i, x_test_split in enumerate(np.array_split(x_test, nb_split)):
        print(f"\tSplit {i+1}/{nb_split}")

        res = model.predict(x_test_split)
        res = res > 0.5
        res = res.astype(int)
        res = np.concatenate((index_split[i].to_numpy().reshape(res.shape[0], 1), res), axis=1)

        np.savetxt(file, res, fmt='%d', delimiter=",")
