from fastapi import HTTPException
from neuralforecast.models import LSTM

from src.core.application.building_model.errors.lstm import LstmTrainSizeError2, LstmTrainSizeError
from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError
from src.core.application.building_model.schemas.lstm import LstmFitResult, LstmParams
from src.infrastructure.factories.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.neural_forecast.base import NeuralForecastInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


class LstmAdapter(NeuralForecastInterface[LstmParams]):
    model_name = "LSTM"
    model = LSTM
    result_class = LstmFitResult

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)

    @staticmethod
    def _validate_params(train_size, val_size, test_size, h, hyperparameters) -> None:
        if h == 0:
            raise HTTPException(
                detail=HorizonValidationError(h=h-test_size, test_size=test_size).detail,
                status_code=400,
            )

        if val_size == 0 and hyperparameters.early_stop_patience_steps > 0:
            raise HTTPException(
                detail=PatienceStepsError().detail,
                status_code=400,
            )

        if val_size != 0 and val_size < h:
            raise HTTPException(
                detail=ValSizeError(val_size=val_size, test_size=test_size, h=h).detail,
                status_code=400,
            )

        if hyperparameters.input_size + h > train_size:
            raise HTTPException(
                status_code=400,
                detail=LstmTrainSizeError(
                    input_size=hyperparameters.input_size,
                    train_size=train_size, h=h, test_size=test_size
                ).detail
            )

        if hyperparameters.recurrent and hyperparameters.input_size + hyperparameters.h_train + test_size > \
                train_size:
            raise HTTPException(
                status_code=400,
                detail=LstmTrainSizeError2(
                    input_size=hyperparameters.input_size,
                    h_train=hyperparameters.h_train,
                    test_size=test_size, train_size=train_size
                ).detail
            )

        return None
