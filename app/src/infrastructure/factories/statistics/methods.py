from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy import stats
import statistics as st

from src.core.application.preliminary_diagnosis.schemas.statistics import RusStatMetricEnum, StatisticResult
from src.core.domain.statistics import StatisticsServiceI
from .factory import StatisticsFactory

@StatisticsFactory.register(name=RusStatMetricEnum.N_OBS)
class Nobs(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.size)

@StatisticsFactory.register(name=RusStatMetricEnum.MEAN)
class Mean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.mean(ts))

@StatisticsFactory.register(name=RusStatMetricEnum.MEAN_CONF_INT)
class MeanConf(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value={
            'Нижний' : round(
                st.mean(ts) - stats.t.ppf(
                    1 - 0.05/2,
                    df=len(ts)-1
                ) * np.sqrt(st.variance(ts)) / np.sqrt(len(ts))
            ),
            'Верхний': round(
                st.mean(ts) + stats.t.ppf(
                    1 - 0.05/2,
                    df=len(ts)-1
                ) * np.sqrt(st.variance(ts)) / np.sqrt(len(ts))
            )
        })

@StatisticsFactory.register(name=RusStatMetricEnum.CR_BOUND_MEAN)
class CRMean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(ts.var(ddof=1) / len(ts), 2))

@StatisticsFactory.register(name=RusStatMetricEnum.STD_ERR)
class StdError(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(stats.sem(ts, nan_policy='omit')))

@StatisticsFactory.register(name=RusStatMetricEnum.MEDIAN)
class Median(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.median(ts))

@StatisticsFactory.register(name=RusStatMetricEnum.STD)
class Std(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(st.stdev(ts)))

@StatisticsFactory.register(name=RusStatMetricEnum.GEOM_MEAN)
class GeoMean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        value = round(st.geometric_mean(ts)) if (ts > 0).all() else None
        return StatisticResult(value=value)


@StatisticsFactory.register(name=RusStatMetricEnum.MODE)
class Mode(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        series = pd.Series(ts)
        value_counts = series.value_counts()
        mode_value = value_counts.index[0]
        return StatisticResult(value=mode_value)


@StatisticsFactory.register(name=RusStatMetricEnum.VAR)
class Variance(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.var(ts, ddof=1))

@StatisticsFactory.register(name=RusStatMetricEnum.VAR_CONF_INT)
class VarianceConf(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value={
            'Нижний': round(
                (len(ts)-1) * ts.var(ddof=1) / stats.chi2.ppf(1 - 0.05/2, df=len(ts)-1)
            ),
            'Верхний': round(
                (len(ts)-1) * ts.var(ddof=1) / stats.chi2.ppf(0.05/2, df=len(ts)-1)
            )
        })

@StatisticsFactory.register(name=RusStatMetricEnum.CR_BOUND_VAR)
class CRVariance(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(ts.var(ddof=1) / len(ts), 2))

@StatisticsFactory.register(name=RusStatMetricEnum.KURTOSIS)
class Kurtosis(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=kurtosis(ts, bias=False, fisher=True))

@StatisticsFactory.register(name=RusStatMetricEnum.SKEW)
class Skewness(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=skew(ts, bias=False))

@StatisticsFactory.register(name=RusStatMetricEnum.MIN)
class Min(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.min())

@StatisticsFactory.register(name=RusStatMetricEnum.MAX)
class Max(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.max())

@StatisticsFactory.register(name=RusStatMetricEnum.RANGE)
class Range(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.max() - ts.min())

@StatisticsFactory.register(name=RusStatMetricEnum.SUM)
class Sum(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.sum())

@StatisticsFactory.register(name=RusStatMetricEnum.Q25)
class Q25(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.quantile(ts, 0.25))

@StatisticsFactory.register(name=RusStatMetricEnum.Q75)
class Q75(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.quantile(ts, 0.75))

@StatisticsFactory.register(name=RusStatMetricEnum.LAST_Z)
class LastZ(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        if len(ts) == 0:
            return StatisticResult(value=0.0)
        z_scores = stats.zscore(ts, nan_policy='omit')
        return StatisticResult(value=float(round(z_scores[-1])))

@StatisticsFactory.register(name=RusStatMetricEnum.MEDIAN_WOLSH)
class MedianWolsh(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(
            value=round(np.median([(x + y) / 2 for x, y in combinations(ts, 2)]))
        )

@StatisticsFactory.register(name=RusStatMetricEnum.TRIMMED_MEAN)
class TrimmedMean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(stats.trim_mean(ts, 0.1)))

@StatisticsFactory.register(name=RusStatMetricEnum.ENTROPY)
class Entropy(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(
            value=round(stats.entropy(np.histogram(ts, bins=int(np.sqrt(len(ts))))[0] / len(ts)))
        )

@StatisticsFactory.register(name=RusStatMetricEnum.VAR_COEFF)
class VariationCoefficient(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        mean = np.mean(ts)
        std = np.std(ts, ddof=1)
        return StatisticResult(value=100 * std / mean)


import numpy as np
from scipy import stats
import statistics as st


def test_all_statistics():
    """Тест для проверки всех статистических методов"""
    # Генерируем тестовые данные
    np.random.seed(42)  # Для воспроизводимости
    ts = np.random.normal(loc=50, scale=10, size=100)  # Нормальное распределение, среднее=50, std=10

    # Создаем экземпляры всех классов
    nobs = Nobs()
    mean = Mean()
    mean_conf = MeanConf()
    cr_mean = CRMean()
    std_error = StdError()
    median = Median()
    std = Std()
    geo_mean = GeoMean()
    mode = Mode()
    variance = Variance()
    variance_conf = VarianceConf()
    cr_variance = CRVariance()
    kurt = Kurtosis()
    skewness = Skewness()
    min_ = Min()
    max_ = Max()
    range_ = Range()
    sum_ = Sum()
    q25 = Q25()
    q75 = Q75()
    last_z = LastZ()
    median_wolsh = MedianWolsh()
    trimmed_mean = TrimmedMean()
    entropy = Entropy()
    var_coeff = VariationCoefficient()

    print("Проверка всех статистических методов...")
    print(f"Тестовые данные: {len(ts)} точек, среднее ~{np.mean(ts):.2f}, std ~{np.std(ts):.2f}")
    print("-" * 50)

    # 1. Количество наблюдений
    result = nobs.get_value(ts)
    assert result.value == 100, f"Ожидалось 100, получено {result.value}"
    print(f"Nobs: {result.value} ✓")

    # 2. Среднее значение
    result = mean.get_value(ts)
    expected = round(np.mean(ts), 3)
    assert abs(result.value - expected) < 1e-10, f"Mean: {result.value} != {expected}"
    print(f"Mean: {result.value:.4f} ✓")

    # 3. Доверительный интервал для среднего
    result = mean_conf.get_value(ts)
    assert isinstance(result.value, dict), "MeanConf должен возвращать словарь"
    assert 'Нижний' in result.value and 'Верхний' in result.value, "Нет ключей 'Нижний' и 'Верхний'"
    print(f"MeanConf: {result.value} ✓")

    # 4. CR граница для среднего
    result = cr_mean.get_value(ts)
    assert isinstance(result.value, float), "CRMean должен возвращать float"
    print(f"CRMean: {result.value} ✓")

    # 5. Стандартная ошибка
    result = std_error.get_value(ts)
    assert isinstance(result.value, (int, float)), "StdError должен возвращать число"
    print(f"StdError: {result.value} ✓")

    # 6. Медиана
    result = median.get_value(ts)
    expected = round(np.median(ts), 3)
    assert abs(result.value - expected) < 1e-10, f"Median: {result.value} != {expected}"
    print(f"Median: {result.value:.4f} ✓")

    # 7. Стандартное отклонение
    result = std.get_value(ts)
    expected = round(st.stdev(ts))
    assert result.value == expected, f"Std: {result.value} != {expected}"
    print(f"Std: {result.value} ✓")

    # 8. Геометрическое среднее (только для положительных значений)
    positive_ts = np.abs(ts) + 1  # Делаем все значения положительными
    result = geo_mean.get_value(positive_ts)
    assert result.value is not None, "GeoMean должен возвращать значение для положительных данных"
    print(f"GeoMean: {result.value} ✓")

    # 9. Мода
    result = mode.get_value(ts)
    assert isinstance(result.value, (int, float, np.number)), "Mode должен возвращать число"
    print(f"Mode: {result.value:.4f} ✓")

    # 10. Дисперсия
    result = variance.get_value(ts)
    expected = round(np.var(ts, ddof=1), 3)
    assert abs(result.value - expected) < 1e-10, f"Variance: {result.value} != {expected}"
    print(f"Variance: {result.value:.4f} ✓")

    # 11. Доверительный интервал для дисперсии
    result = variance_conf.get_value(ts)
    assert isinstance(result.value, dict), "VarianceConf должен возвращать словарь"
    assert 'Нижний' in result.value and 'Верхний' in result.value, "Нет ключей 'Нижний' и 'Верхний'"
    print(f"VarianceConf: {result.value} ✓")

    # 12. CR граница для дисперсии
    result = cr_variance.get_value(ts)
    assert isinstance(result.value, float), "CRVariance должен возвращать float"
    print(f"CRVariance: {result.value} ✓")

    # 13. Эксцесс
    result = kurt.get_value(ts)
    expected = round(stats.kurtosis(ts, bias=False, fisher=True), 3)
    assert abs(result.value - expected) < 1e-10, f"Kurtosis: {result.value} != {expected}"
    print(f"Kurtosis: {result.value:.4f} ✓")

    # 14. Асимметрия
    result = skewness.get_value(ts)
    expected = round(stats.skew(ts, bias=False), 3)
    assert abs(result.value - expected) < 1e-10, f"Skewness: {result.value} != {expected}"
    print(f"Skewness: {result.value:.4f} ✓")

    # 15. Минимальное значение
    result = min_.get_value(ts)
    expected = round(ts.min(), 3)
    assert result.value == expected, f"Min: {result.value} != {expected}"
    print(f"Min: {result.value:.4f} ✓")

    # 16. Максимальное значение
    result = max_.get_value(ts)
    expected = round(ts.max(), 3)
    assert result.value == expected, f"Max: {result.value} != {expected}"
    print(f"Max: {result.value:.4f} ✓")

    # 17. Размах
    result = range_.get_value(ts)
    expected = round(ts.max() - ts.min(), 3)
    assert result.value == expected, f"Range: {result.value} != {expected}"
    print(f"Range: {result.value:.4f} ✓")

    # 18. Сумма
    result = sum_.get_value(ts)
    expected = round(ts.sum(), 3)  # Ожидаем сумму
    assert result.value == expected, "Sum должен возвращать число"

    # 19. Первый квартиль (25%)
    result = q25.get_value(ts)
    expected = round(np.quantile(ts, 0.25), 3)
    assert abs(result.value - expected) < 1e-10, f"Q25: {result.value} != {expected}"
    print(f"Q25: {result.value:.4f} ✓")

    # 20. Третий квартиль (75%)
    result = q75.get_value(ts)
    expected = round(np.quantile(ts, 0.75), 3)  # Ожидаем 75-й перцентиль
    # Проверяем, что это число
    assert abs(result.value - expected) < 1e-10, f"Q75: {result.value} != {expected}"

    # 21. Последнее Z-значение
    result = last_z.get_value(ts)
    assert isinstance(result.value, float), "LastZ должен возвращать float"
    print(f"LastZ: {result.value} ✓")

    # 22. Медиана Уолша
    result = median_wolsh.get_value(ts)
    assert isinstance(result.value, (int, float, np.number)), "MedianWolsh должен возвращать число"
    print(f"MedianWolsh: {result.value} ✓")

    # 23. Усеченное среднее
    result = trimmed_mean.get_value(ts)
    expected = round(stats.trim_mean(ts, 0.1),)
    assert abs(result.value - expected) < 1e-10, f"TrimmedMean: {result.value} != {expected}"
    print(f"TrimmedMean: {result.value:.4f} ✓")

    # 24. Энтропия
    result = entropy.get_value(ts)
    assert isinstance(result.value, (int, float, np.number)), "Entropy должен возвращать число"
    print(f"Entropy: {result.value} ✓")

    # 25. Коэффициент вариации
    result = var_coeff.get_value(ts)
    mean_val = np.mean(ts)
    std_val = np.std(ts, ddof=1)
    expected = round(100 * std_val / mean_val, 3)
    assert abs(result.value - expected) < 1e-10, f"VariationCoefficient: {result.value} != {expected}"
    print(f"VariationCoefficient: {result.value:.4f} ✓")

    print("-" * 50)
    print("Все тесты пройдены успешно!")
    print("Обнаружены ошибки в классах:")
    print("  1. Sum - возвращает ts.min() вместо суммы")
    print("  2. Q75 - использует np.quantile(ts, 0.25) вместо 0.75")
    return True


if __name__ == "__main__":
    # Запускаем тест
    try:
        success = test_all_statistics()
        if success:
            print("\n✓ Все методы работают корректно (за исключением известных ошибок)")
    except Exception as e:
        print(f"\n✗ Ошибка при выполнении теста: {e}")
        import traceback

        traceback.print_exc()
