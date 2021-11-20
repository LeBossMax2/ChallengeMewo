import pandas as pd
import numpy as np

x_data = pd.read_csv("../train_X.csv", index_col=0, sep=',')
y_data = pd.read_csv("../train_y.csv", index_col=0, sep=',')

nb_split = 10000

index_split = np.array_split(x_data.index, nb_split)

file_x = open("../train_X_augmented.csv", "w")
file_x.write("ChallengeID,")
np.savetxt(file_x, x_data.columns.to_numpy().reshape(1, len(x_data.columns)), fmt='%s', delimiter=",")
file_y = open("../train_y_augmented.csv", "w")
file_y.write("ChallengeID,")
np.savetxt(file_y, y_data.columns.to_numpy().reshape(1, len(y_data.columns)), fmt='%s', delimiter=",")

index_split = np.array_split(x_data.index, nb_split)
splits_x = np.array_split(x_data, nb_split)
splits_y = np.array_split(y_data, nb_split)

for i in range(nb_split):
    print(f"\tSplit {i+1}/{nb_split}")
    split_x = splits_x[i]
    split_y = splits_y[i]
    
    res_x = split_x.iloc[:,range(0,289)]
    res_y = split_y.iloc[:,range(0,214)]

    res_x = np.concatenate((index_split[i].to_numpy().reshape(res_x.shape[0], 1), res_x), axis=1)
    res_y = np.concatenate((index_split[i].to_numpy().reshape(res_y.shape[0], 1), res_y), axis=1)

    np.savetxt(file_x, res_x, fmt="%s", delimiter=",")
    np.savetxt(file_y, res_y, fmt='%s', delimiter=",")


    res_x = split_x.iloc[:,range(0,289)]
    res_y = split_y.iloc[:,range(0,214)]
    x_shape = res_x.shape
    res_x = res_x + np.random.normal(0, 0.0005, x_shape[0]*x_shape[1]).reshape(x_shape)
    res_x = np.where(res_x<0, 0, res_x)
    res_x = np.where(res_x>1, 1, res_x)

    res_x = np.concatenate((index_split[i].to_numpy().reshape(res_x.shape[0], 1)+110855, res_x), axis=1)
    res_y = np.concatenate((index_split[i].to_numpy().reshape(res_y.shape[0], 1)+110855, res_y), axis=1)

    np.savetxt(file_x, res_x, fmt='%s', delimiter=",")
    np.savetxt(file_y, res_y, fmt='%s', delimiter=",")