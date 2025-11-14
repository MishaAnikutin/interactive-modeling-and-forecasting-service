from abc import ABC
import numpy as np

from src.core.domain.distributions import CDF, PDF
from src.core.domain.distributions.density_estimation import PPF


# TODO: подумать, есть ли проблема что он не падает на этапе сборки если класс не соответствует контракту
class DistributionServiceI(ABC):
    """
    Класс для расчета теоретических вероятностей по заданной функции распределения

    Нужно либо указать impl, либо перегрузить get_theoretical_probs
    """
    impl = None

    def get_cdf(self, x: np.array) -> CDF:
        params = self.impl.fit(x)

        theoretical_probs = self.impl.cdf(x, *params)
        y = theoretical_probs.tolist()

        return CDF(x=x, y=y)

    def get_ppf(self, x: np.array) -> PPF:
        params = self.impl.fit(x)

        n = len(x)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_probs = self.impl.ppf(probs, *params)
        y = theoretical_probs.tolist()

        return PPF(x=x, y=y)

    def get_pdf(self, x: np.array) -> PDF:
        params = self.impl.fit(x)

        theoretical_probs = self.impl.pdf(x, *params)
        y = theoretical_probs.tolist()

        return PDF(x=x, y=y)
