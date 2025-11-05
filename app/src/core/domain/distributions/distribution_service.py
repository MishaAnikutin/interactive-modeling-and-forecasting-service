from abc import ABC
import numpy as np

from src.core.domain.distributions import CDF, PDF


# TODO: подумать, есть ли проблема что он не падает на этапе сборки если класс не соответствует контракту
class DistributionServiceI(ABC):
    """
    Класс для расчета теоретических вероятностей по заданной функции распределения

    Нужно либо указать impl либо перегрузить get_theoretical_probs
    """
    impl = None

    def get_cdf(self, x: np.array, must_sort: bool) -> CDF:
        params = self.impl.fit(x)

        if must_sort:
            x = np.sort(x)

        theoretical_probs = self.impl.cdf(x, *params)

        y = theoretical_probs.tolist()

        return CDF(x=x, y=y)

    def get_pdf(self, x: np.array, must_sort: bool) -> PDF:
        params = self.impl.fit(x)

        if must_sort:
            x = np.sort(x)

        theoretical_probs = self.impl.pdf(x, *params)

        y = theoretical_probs.tolist()

        return PDF(x=x, y=y)
