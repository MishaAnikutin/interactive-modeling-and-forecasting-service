from abc import ABC
import numpy as np


# TODO: подумать, есть ли проблема что он не падает на этапе сборки если класс не соответствует контракту
class DistributionServiceI(ABC):
    """
    Класс для расчета теоретических вероятностей по заданной функции распределения

    Нужно либо указать impl либо перегрузить get_theoretical_probs
    """
    impl = None

    def get_cdf(self, x: np.array, must_sort: bool) -> list[float]:
        params = self.impl.fit(x)

        if must_sort:
            x = np.sort(x)

        theoretical_probs = self.impl.cdf(x, *params)

        return theoretical_probs.tolist()

    def get_pdf(self, x: np.array, must_sort: bool) -> list[float]:
        params = self.impl.fit(x)

        if must_sort:
            x = np.sort(x)

        theoretical_probs = self.impl.pdf(x, *params)

        return theoretical_probs.tolist()
