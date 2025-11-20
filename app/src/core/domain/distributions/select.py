from enum import StrEnum

from pydantic import BaseModel


class SelectDistributionStatistics(StrEnum):
    RSS: str = 'RSS'
    wasserstein: str = 'wasserstein'
    ks: str = 'ks'
    energy: str = 'energy'
    goodness_of_fit: str = 'goodness_of_fit'


class SelectDistributionMethod(StrEnum):
    parametric: str = 'parametric'
    quantile: str = 'quantile'
    percentile: str = 'percentile'
    discrete: str = 'discrete'



