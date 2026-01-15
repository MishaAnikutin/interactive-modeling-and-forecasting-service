from enum import Enum


class InformationCriteriaScoring(str, Enum):
    aic: str = 'aic'
    bic: str = 'bic'
