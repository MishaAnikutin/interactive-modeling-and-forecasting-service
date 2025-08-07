import pandas as pd
from typing import List
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from logs import logger

from src.core.application.building_model.schemas.arimax import ArimaxParams, ArimaxFitResult
# FIXME: схемки это конечно ничего, но получается что инфра зависит от слоя приложения

from src.infrastructure.adapters.modeling.interface import MlAdapterInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
from src.infrastructure.adapters.metrics import MetricsFactory

from src.core.domain import FitParams, Coefficient, DataFrequency


class ArimaxAdapter(MlAdapterInterface):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
            # FIXME: кстати, а не нарушается ли SRP? этот класс и делит выборки и обучает модель
    ):
        super().__init__(metric_factory, ts_train_test_split)
        self._log = logger.getChild(self.__class__.__name__)
        self._ts_spliter = ts_train_test_split  # Сохраняем для удобства

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            arimax_params: ArimaxParams,
            fit_params: FitParams,
            data_frequency: DataFrequency,
    ) -> tuple[ArimaxFitResult, SARIMAXResultsWrapper]:
        self._log.debug("Старт обучения ARIMAX")

        # 1. Разделение данных -------------------------------------------------------
        (
            exog_train,
            train_target,
            exog_val,
            val_target,
            exog_test,
            test_target,
        ) = self._ts_spliter.split(
            train_boundary=fit_params.train_boundary,
            val_boundary=fit_params.val_boundary,
            target=target,
            exog=exog,
        )

        # 2. Создаём и обучаем модель ------------------------------------------------
        # Используем SARIMAX вместо ARIMA для поддержки метода apply()
        model = sm.tsa.SARIMAX(
            endog=train_target,
            exog=exog_train,
            order=(arimax_params.p, arimax_params.d, arimax_params.q),
            seasonal_order=(0, 0, 0, 0),  # Без сезонности
            trend='c'  # Константа
        )
        results = model.fit(disp=False)
        self._log.info("Модель обучена", extra={"aic": results.aic, "bic": results.bic})

        # 3. Прогнозы ----------------------------------------------------------------
        # 3.1 In-sample прогноз для тренировочных данных
        train_predict = results.get_prediction().predicted_mean

        # 3.2 Прогноз для валидации с использованием фактических лагов
        val_predict = None
        if val_target is not None:
            # Объединяем train + val
            full_val_target = pd.concat([train_target, val_target])
            full_val_exog = pd.concat([exog_train, exog_val]) if exog is not None else None

            # Применяем модель к объединенным данным
            val_model = results.apply(full_val_target, exog=full_val_exog)
            # Получаем предсказания только для валидационного периода
            val_predict = val_model.get_prediction(
                start=val_target.index[0],
                end=val_target.index[-1]
            ).predicted_mean

        # 3.3 Прогноз для теста с использованием фактических лагов
        test_predict = None
        if test_target is not None:
            # Объединяем train + val (если есть) + test
            full_test_target = train_target
            full_test_exog = exog_train if exog is not None else None

            if val_target is not None:
                full_test_target = pd.concat([full_test_target, val_target])
                full_test_exog = pd.concat([full_test_exog, exog_val]) if exog is not None else None

            full_test_target = pd.concat([full_test_target, test_target])
            full_test_exog = pd.concat([full_test_exog, exog_test]) if exog is not None else None

            # Применяем модель к объединенным данным
            test_model = results.apply(full_test_target, exog=full_test_exog)
            # Получаем предсказания только для тестового периода
            test_predict = test_model.get_prediction(
                start=test_target.index[0],
                end=test_target.index[-1]
            ).predicted_mean

        # 3.4 Out-of-sample прогноз (рекурсивный)
        forecast = pd.Series()
        if exog is None:  # Только если нет экзогенных переменных
            forecast = results.get_forecast(
                steps=fit_params.forecast_horizon
            ).predicted_mean

        # 4. Формируем объект Forecasts ----------------------------------------------
        forecasts = self._generate_forecasts(
            train_predict=train_predict,
            validation_predict=val_predict,
            test_predict=test_predict,
            forecast=forecast,
            data_frequency=data_frequency
        )

        # 5. Метрики -----------------------------------------------------------------
        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_val_true=val_target,
            y_val_pred=val_predict,
            y_test_true=test_target,
            y_test_pred=test_predict,
        )

        # 6. Коэффициенты ------------------------------------------------------------
        coefficients = self._parse_coefficients(results)

        # 7. Результат ---------------------------------------------------------------
        result = ArimaxFitResult(
            coefficients=coefficients,
            model_metrics=metrics,
            forecasts=forecasts
        )
        return result, results

    def _parse_coefficients(self, results) -> List[Coefficient]:
        return [
            Coefficient(name=name, value=value, p_value=results.pvalues[name])
            for name, value in results.params.items()
        ]