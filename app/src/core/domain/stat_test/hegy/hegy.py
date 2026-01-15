from enum import Enum


class TrendType(str, Enum):
    NONE = "n"
    CONSTANT = "c"
    CONSTANT_SEASONAL_DUMMIES = "cd"
    CONSTANT_TREND = "ct"
    CONSTANT_SEASONAL_TREND = "cdt"


class CriteriaType(str, Enum):
    AIC = "aic"
    BIC = "bic"
    HQIC = "hqic"

