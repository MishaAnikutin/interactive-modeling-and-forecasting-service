import numpy as np

from src.core.domain import Timeseries


def get_residuals(y_true: Timeseries, y_pred: Timeseries) -> np.array:
    y_true = np.array(y_true.values)
    y_pred = np.array(y_pred.values)
    return y_true - y_pred