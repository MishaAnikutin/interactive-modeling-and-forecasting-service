from enum import Enum


class VarianceEstimator(str, Enum):
    acf: str = "acf"
    bartlett: str = "bartlett"
