import pandas as pd
from fastapi import HTTPException
from neuralforecast.models import NHITS

from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError, \
    TrainSizeError
from src.core.application.building_model.schemas.nhits import NhitsParams, NhitsFitResult
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.neural_forecast.base import NeuralForecastInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


class NhitsAdapter(NeuralForecastInterface[NhitsParams]):
    model_name = "NHITS"
    model = NHITS
    result_class = NhitsFitResult

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)

    def _get_model(self, exog: pd.DataFrame | None, hyperparameters: NhitsParams, h: int):
        model = self.model(
            input_size=3 * h,
            hist_exog_list=[exog_col for exog_col in exog.columns] if exog is not None else None,
            accelerator='cpu',
            h=h,
            **hyperparameters.model_dump()
        )
        return model

    @staticmethod
    def _validate_params(train_size, val_size, test_size, h, hyperparameters) -> None:
        if h == 0:
            raise HTTPException(
                detail=HorizonValidationError().detail,
                status_code=400,
            )

        if val_size != 0 and val_size < h:
            raise HTTPException(
                detail=ValSizeError().detail,
                status_code=400,
            )

        if val_size == 0 and hyperparameters.early_stop_patience_steps > 0:
            raise HTTPException(
                detail=PatienceStepsError().detail,
                status_code=400,
            )

        if 4 * h > train_size:
            raise HTTPException(
                status_code=400,
                detail=TrainSizeError().detail
            )

        return None
