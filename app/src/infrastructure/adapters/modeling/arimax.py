import pandas as pd
from typing import List
import statsmodels.api as sm
from logs import logger

from src.core.domain import FitParams, Coefficient
from src.core.application.building_model.schemas.arimax import ArimaxParams, ArimaxFitResult
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface

from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
from src.infrastructure.adapters.metrics import MetricsFactory


class ArimaxAdapter(MlAdapterInterface):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
        self,
        metric_factory: MetricsFactory,
        ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)
        self._log = logger.getChild(self.__class__.__name__)

    def fit(
        self,
        target: pd.Series,
        exog: pd.DataFrame | None,
        arimax_params: ArimaxParams,
        fit_params: FitParams,
    ) -> ArimaxFitResult:
        self._log.debug("Старт обучения ARIMAX")

        # 1. Train / val / test split -------------------------------------------------
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
        # 2. Подготовка данных --------------------------------------------------------
        # 3. Создаём и обучаем модель -------------------------------------------------
        model = sm.tsa.ARIMA(
            endog=train_target,
            exog=exog_train,
            order=(arimax_params.p, arimax_params.d, arimax_params.q),
        )
        results = model.fit()

        self._log.info("Модель обучена", extra={"aic": results.aic, "bic": results.bic})

        # 4. Прогнозы -----------------------------------------------------------------
        # 4.1 train
        train_predict = results.get_prediction().predicted_mean

        # 4.2 test
        test_predict = results.get_forecast(
            steps=len(test_target), exog=exog_test
        ).predicted_mean

        # 4.3 out-of-sample прогноз (если надо)
        forecast = (
            None
            if exog is not None
            else results.forecast(steps=fit_params.forecast_horizon)
        )

        # ---------- формируем объект Forecasts -----------------------------------
        forecasts = self._generate_forecasts(
            train_predict=train_predict,
            test_predict=test_predict,
            forecast=forecast
        )

        # 5. Метрики ------------------------------------------------------------------
        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_test_true=test_target,
            y_test_pred=test_predict,
        )

        coefficients = self._parse_coefficients(results)

        # 6. Результат ----------------------------------------------------------------
        return ArimaxFitResult(
            coefficients=coefficients,
            model_metrics=metrics,
            forecasts=forecasts,
            weight_path=fit_params.weight_path,
            model_id=fit_params.model_id,
        )

    def _parse_coefficients(self, results) -> List[Coefficient]:
        return [
            Coefficient(name=name, value=value, p_value=results.pvalues[name])
            for name, value in results.params.items()
        ]
