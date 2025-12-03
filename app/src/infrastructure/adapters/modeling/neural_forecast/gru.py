from fastapi import HTTPException
from neuralforecast.models import GRU

from src.core.application.building_model.errors.lstm import LstmTrainSizeError2, LstmTrainSizeError
from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError
from src.core.application.building_model.schemas.gru import GruParams, GruFitResult
from src.infrastructure.factories.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.neural_forecast.base import NeuralForecastInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


class GruAdapter(NeuralForecastInterface[GruParams]):
    model_name = "GRU"
    model = GRU
    result_class = GruFitResult

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)

    @staticmethod
    def _validate_params(train_size, val_size, test_size, h, hyperparameters) -> None:
        set_automatically = False
        if hyperparameters.input_size == -1:
            set_automatically = True
            hyperparameters.input_size = 3 * h

        if h == 0:
            raise HTTPException(
                detail=HorizonValidationError(h=h-test_size, test_size=test_size).detail,
                status_code=400,
            )

        if val_size != 0 and val_size < h:
            raise HTTPException(
                detail=ValSizeError(
                    val_size=val_size,
                    test_size=test_size,
                    h=h
                ).detail,
                status_code=400,
            )

        if val_size == 0 and hyperparameters.early_stop_patience_steps > 0:
            raise HTTPException(
                detail=PatienceStepsError().detail,
                status_code=400,
            )

        if hyperparameters.input_size + h > train_size:
            raise HTTPException(
                status_code=400,
                detail=LstmTrainSizeError(
                    input_size=hyperparameters.input_size,
                    train_size=train_size, h=h, test_size=test_size
                ).detail + f". input_size выставлен автоматически в значение {hyperparameters.input_size} =(3 * h), "
                           f"так как пользователь указал значение -1." if set_automatically else "",
            )

        if hyperparameters.recurrent and hyperparameters.input_size + hyperparameters.h_train + test_size > \
                train_size:
            raise HTTPException(
                status_code=400,
                detail=LstmTrainSizeError2(
                    input_size=hyperparameters.input_size,
                    h_train=hyperparameters.h_train,
                    test_size=test_size, train_size=train_size
                ).detail + f". input_size выставлен автоматически в значение {hyperparameters.input_size} =(3 * h), "
                           f"так как пользователь указал значение -1." if set_automatically else ""
            )

        return None
