from enum import Enum


class LossFunction(str, Enum):
    MSE = "mse"
    MAE = "mae"
