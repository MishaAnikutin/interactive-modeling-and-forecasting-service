import uuid

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from logs import logger
from src.core.application.building_model.schemas.nhits import (
    NhitsParams,
    NhitsFitResult,
)

from src.core.domain import FitParams
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class NhitsAdapter(MlAdapterInterface):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)
        self._log = logger.getChild(self.__class__.__name__)

    @staticmethod
    def _to_panel(
        target: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "unique_id": 'ts',
                "ds": target.index,
                "y": target.values,
            }
        )
        if exog is not None and not exog.empty:
            raise NotImplementedError("Необходимо реализовать логику построения данных с экз переменными")
        return df

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            nhits_params: NhitsParams,
            fit_params: FitParams,
    ) -> NhitsFitResult:
        self._log.debug("Старт обучения NHiTS")

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
        train_df = self._to_panel(
            target=pd.concat([train_target, val_target]),
            exog=None
        )
        val_size = val_target.shape[0]
        test_size = test_target.shape[0]

        future_df = self._to_panel(
            target=test_target,
            exog=None
        )

        # 3. Создаём и обучаем модель -------------------------------------------------
        model = NHITS(
            accelerator='cpu',
            h=test_size,
            input_size=test_size*3,
            **nhits_params.model_dump()
        )
        nf = NeuralForecast(models=[model], freq="ME")

        nf.fit(df=train_df, val_size=val_size)
        self._log.info("Модель NHiTS обучена")

        # 4. Прогнозы -----------------------------------------------------------------
        # 4.1 train
        fcst_insample_df = nf.predict_insample()
        fcst_train = fcst_insample_df.drop_duplicates(subset="ds", keep="last").set_index("ds")['NHITS']

        # 4.2 test
        forecasts = nf.predict(futr_df=future_df)
        fcst_test = forecasts['NHITS']

        # 4.3 future
        fcst_future = nf.predict().set_index("ds")['NHITS']

        # ---------- формируем объект Forecasts -----------------------------------
        forecasts = self._generate_forecasts(
            train_predict=fcst_train,
            test_predict=fcst_test,
            forecast=fcst_future,
        )

        # 5. Метрики ------------------------------------------------------------------
        metrics = self._calculate_metrics(
            y_train_true=train_df['y'],
            y_train_pred=fcst_train,
            y_test_true=test_target,
            y_test_pred=fcst_test,
        )

        # 6. Результат ----------------------------------------------------------------
        return NhitsFitResult(
            forecasts=forecasts,
            model_metrics=metrics,
            weight_path=str('заглушка'),
            model_id=str(uuid.uuid4()),
        )