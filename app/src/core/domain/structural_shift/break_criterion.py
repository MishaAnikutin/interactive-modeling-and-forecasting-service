from enum import Enum


class BreakCriterion(str, Enum):
    ssr: str = 'ssr'
    aic: str = 'aic'
    bic: str = 'bic'
    rsquared: str = 'rsquared'
    rsquared_adj: str = 'rsquared_adj'
