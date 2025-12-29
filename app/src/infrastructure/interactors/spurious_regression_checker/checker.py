import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


class SpuriousRegressionChecker:
    def __init__(self, r2_threshold=0.2, dw_threshold=0.1):
        self.r2_threshold = r2_threshold
        self.dw_threshold = dw_threshold

    def check(self, y, X, add_constant=True):
        """
        Проверка на ложную регрессию

        Parameters:
        -----------
        y : array-like
            Зависимая переменная
        X : array-like
            Независимые переменные
        add_constant : bool
            Добавлять ли константу в регрессию

        Returns:
        --------
        dict с результатами проверки
        """
        # Подготовка данных
        if add_constant:
            X = sm.add_constant(X)

        # Оценка модели
        model = sm.OLS(y, X).fit()

        # Расчет статистик
        r_squared = model.rsquared
        dw_stat = durbin_watson(model.resid)

        # Проверка критериев
        is_spurious = (r_squared > self.r2_threshold) and (dw_stat < self.dw_threshold)

        # Определение уровня предупреждения
        if is_spurious:
            warning_level = "high"
        elif r_squared > self.r2_threshold * 0.7:
            warning_level = "medium"
        else:
            warning_level = "none"

        # Формирование рекомендаций
        recommendations = []
        if is_spurious:
            recommendations.extend([
                "Обнаружены признаки ложной регрессии",
                "Рекомендуется проверить ряды на стационарность",
                "Рассмотрите использование модели с поправкой на автокорреляцию",
                "Или используйте разности рядов вместо уровней"
            ])

        return {
            "is_spurious": is_spurious,
            "warning_level": warning_level,
            "metrics": {
                "r_squared": round(r_squared, 4),
                "durbin_watson": round(dw_stat, 4),
                "significant_coefs": sum(model.pvalues < 0.05),
                "aic": round(model.aic, 2),
                "bic": round(model.bic, 2)
            },
            "recommendations": recommendations,
            "model_summary": str(model.summary())  # Для детального анализа
        }
