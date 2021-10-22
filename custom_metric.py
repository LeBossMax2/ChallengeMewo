"""
Example of custom metric script.
The custom metric script must contain the definition of custom_metric_function and a main function
that reads the two csv files with pandas and evaluate the custom metric.
"""

import numpy as np
from sklearn.metrics import f1_score

def custom_metric_function(dataframe_y_true, dataframe_y_pred):
    """
        Example of custom metric function.
        NOTA: the order (dataframe_y_true, dataframe_y_pred) matters if the metric is
        non symmetric.

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """

    score = f1_score(dataframe_y_true, dataframe_y_pred, average="weighted")

    return score


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_Y_TRUE = '---.csv'
    CSV_FILE_Y_PRED = '---.csv'
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    print(custom_metric_function(df_y_true, df_y_pred))
