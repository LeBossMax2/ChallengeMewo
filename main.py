import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from custom_metric import custom_metric_function

x_data_init = pd.read_csv("../train_X.csv", index_col=0, sep=',')
y_data = pd.read_csv("../train_y.csv", index_col=0, sep=',')

x_data = []
for rng in [range(0, 90), range(90, 202), range(202, 248), range(248, 266), range(266, 281), range(281, 289)]:
    x_data += [x_data_init.iloc[:, rng]]

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_data, y_data)

custom_loss = tf.keras.losses.BinaryCrossentropy()
custom_opt = tf.keras.optimizers.Adam(learning_rate=0.01)

input_layers = [
    Input(shape = (90,)),
    Input(shape = (112,)),
    Input(shape = (46,)),
    Input(shape = (18,)),
    Input(shape = (15,)),
    Input(shape = (8,))
]

first_part = []

for i in range(0, 3):
    layer = Concatenate()([input_layers[i], input_layers[i + 3]])
    layer = Dense(150, activation="relu")(layer)
    first_part += [Dense(150, activation="relu")(layer)]

layer = Concatenate()(first_part)
layer = Dense(280, activation="relu")(layer)
second_part = Dense(280, activation="relu")(layer)

last_part = []

for i in range(0, 3):
    layer = Dense(150, activation="relu")(second_part)
    layer = Concatenate()([layer, input_layers[i]])
    last_part += [Dense(input_layers[i].shape[1], activation="sigmoid")(layer)]

model = Model(inputs = input_layers, outputs = Concatenate()(last_part))

model.compile(loss=custom_loss, optimizer=custom_opt, metrics="mae")

model.summary()

batch_size = 10
epochs = 8

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
