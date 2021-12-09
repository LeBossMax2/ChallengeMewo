import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from custom_metric import custom_metric_function

def splitter(data):
    x_data = []
    for rng in [range(0, 90), range(90, 202), range(202, 248), range(248, 266), range(266, 281), range(281, 289)]:
        x_data += [data.iloc[:, rng]]
    return x_data

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    return 1 - f1_loss(y_true, y_pred)

# Function inspired by https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
def f1_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float')
    y_pred = K.cast(y_pred, 'float')

    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def weighted_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    return 1 - weighted_f1_loss(y_true, y_pred)

def partial_weighted_f1(slice, name):
    def metric(y_true, y_pred):
        return weighted_f1(y_true[:, slice], y_pred[:, slice])
    metric.__name__ = 'wf1_' + name
    return metric

def weighted_f1_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float')
    y_pred = K.cast(y_pred, 'float')

    gp = K.sum(y_true, axis=0)
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    weighted_f1 = f1 * gp / K.sum(gp) 
    weighted_f1 = tf.where(tf.math.is_nan(weighted_f1), tf.zeros_like(weighted_f1), weighted_f1)
    return 1 - K.sum(weighted_f1)

def wf1_loss_p(y_true, y_pred):
    return (weighted_f1_loss(y_true[slice(0, 90)], y_pred[slice(0, 90)])
          + weighted_f1_loss(y_true[slice(90, 202)], y_pred[slice(90, 202)])
          + weighted_f1_loss(y_true[slice(202, 248)], y_pred[slice(202, 248)])) / 3.0

# Read train data
x_data_init = pd.read_csv("../train_X.csv", index_col=0, sep=',')
y_data = pd.read_csv("../train_y.csv", index_col=0, sep=',')

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_data_init, y_data)

x_train = splitter(x_train)
x_valid = splitter(x_valid)

# Create model
custom_loss = wf1_loss_p
custom_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
metrics=["mae", "accuracy", weighted_f1, f1,
    partial_weighted_f1(slice(0, 90), "genres"), partial_weighted_f1(slice(90, 202), "instruments"), partial_weighted_f1(slice(202, 248), "moods")]

input_layers = [
    Input(shape = (90,)),
    Input(shape = (112,)),
    Input(shape = (46,)),
    Input(shape = (18,)),
    Input(shape = (15,)),
    Input(shape = (8,))
]

def block(inputs, act):
    first_part = []

    for i in range(0, 3):
        layer = Concatenate()([inputs[i], inputs[i + 3]])
        #layer = Dense(150, activation="relu")(layer)
        first_part += [Dense(int((inputs[i].shape[1] + inputs[i + 3].shape[1]) * 1.2), activation="relu")(layer)]

    layer = Concatenate()(first_part)
    layer = Dense(400, activation="relu")(layer)
    second_part = Dropout(0.1)(layer)

    last_part = []

    for i in range(0, 3):
        layer = Dense(inputs[i].shape[1])(second_part)
        layer = Add()([inputs[i], layer])
        last_part += [Activation(act)(layer)]
    
    return last_part

b = block(input_layers, "relu")
#b = block(b + input_layers[3:], "relu")
b = block(b + input_layers[3:], "sigmoid")

model = Model(inputs = input_layers, outputs = Concatenate()(b))

model.compile(loss=custom_loss, optimizer=custom_opt, metrics=metrics)

model.summary()

# Train model
batch_size = 512
epochs = 600

hist = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_valid, y_valid),
        callbacks=
        [
            EarlyStopping(
                monitor='val_loss', min_delta=0, patience=8, verbose=0, restore_best_weights=True
            )
        ])

model.save_weights("weights")

# Print results
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Val loss:', score[0])
print('Val metrics:', score[1:])

# Show loss history
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Compute test prediction
x_test = pd.read_csv("../test_X.csv", index_col=0, sep=',')

nb_split = 100

index_split = np.array_split(x_test.index, nb_split)

with open("../pred_y.csv", "w") as file:
    file.write("ChallengeID,")
    np.savetxt(file, y_data.columns.to_numpy().reshape(1, len(y_data.columns)), fmt='%s', delimiter=",")

    for i, x_test_split in enumerate(np.array_split(x_test, nb_split)):
        x_test_split = splitter(x_test_split)
        print(f"\tSplit {i+1}/{nb_split}")

        res = model.predict(x_test_split)
        res = res > 0.5
        res = res.astype(int)
        res = np.concatenate((index_split[i].to_numpy().reshape(res.shape[0], 1), res), axis=1)

        np.savetxt(file, res, fmt='%d', delimiter=",")
