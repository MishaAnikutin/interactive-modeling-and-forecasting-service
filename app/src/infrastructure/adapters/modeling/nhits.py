import pandas as pd

from logs import logger
from src.core.application.building_model.schemas.nhits import NhitsFitResult, NhitsParams
from src.core.domain import FitParams
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit

from neuralforecast.models import NHITS

class NhitsAdapter:
    metrics = ()

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split
        self._log = logger.getChild(self.__class__.__name__)

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            nhits_params: NhitsParams,
            fit_params: FitParams,
    ) -> NhitsFitResult:
        self._log.debug(
            "Старт обучения NHiTS",
        )

        # Делим выборку на train / val / test
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

        # Создаем и обучаем модель
        ...

        self._log.info("Модель NHiTS обучена")

        # Строим прогноз на обучающей выборке
        ...

        # Строим прогноз по тестовой выборке
        ...

        # Строим вневыборочный прогноз (если нет экзогенных переменных)
        ...