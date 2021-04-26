import numpy as np
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import accuracy_score


def rmsle(y_true, y_prediction):
    return np.sqrt(msle(y_true, y_prediction))
