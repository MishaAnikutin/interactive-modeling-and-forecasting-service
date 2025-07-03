from datetime import datetime

import pytest

from src.core.application.building_model.schemas.nhits import NhitsParams
from src.core.domain import FitParams, DataFrequency
from tests.conftest import nhits_adapter, ipp_eu

@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "nhits_params, fit_params",
    [
        # Базовый сценарий (уже был)
        (
            NhitsParams(
                max_steps=30,
                early_stop_patience_steps=3,
                val_check_steps=50,
                learning_rate=1e-3,
                scaler_type="robust",
            ),
            FitParams(
                train_boundary=datetime(2016, 6, 30),
                val_boundary=datetime(2018, 5, 31),
                forecast_horizon=20,
                data_frequency=DataFrequency.month,
            ),
        ),
        # ─────────────── Дополнительные граничные кейсы ───────────────
        # # 1. Минимально допустимые значения по шагам + очень маленький learning_rate
        # (
        #     NhitsParams(
        #         max_steps=1,
        #         early_stop_patience_steps=0,
        #         val_check_steps=0,
        #         learning_rate=1e-6,
        #         scaler_type="standard",
        #     ),
        #     FitParams(
        #         train_boundary=datetime(2015, 12, 31),
        #         val_boundary=datetime(2016, 1, 31),
        #         forecast_horizon=48,
        #         data_frequency=DataFrequency.month,
        #     ),
        # ),
        # # 2. early_stop_patience_steps больше max_steps, val_check_steps = 1, большой learning_rate
        # (
        #     NhitsParams(
        #         max_steps=5,
        #         early_stop_patience_steps=10,
        #         val_check_steps=1,
        #         learning_rate=1e-2,
        #         scaler_type="robust",
        #     ),
        #     FitParams(
        #         train_boundary=datetime(2016, 6, 30),
        #         val_boundary=datetime(2017, 6, 30),
        #         forecast_horizon=10,
        #         data_frequency=DataFrequency.month,
        #     ),
        # ),
        # 3. Очень длинное обучение + большой patience + длительный горизонт прогноза
        (
            NhitsParams(
                max_steps=1_000,
                early_stop_patience_steps=50,
                val_check_steps=200,
                learning_rate=5e-4,
                scaler_type="robust",
            ),
            FitParams(
                train_boundary=datetime(2013, 12, 31),
                val_boundary=datetime(2017, 12, 31),
                forecast_horizon=36,
                data_frequency=DataFrequency.month,
            ),
        ),
    ]
)
def test_nhits_adapter_fit_without_exog_and_with_month_ending_data(
    nhits_params,
    fit_params,
    nhits_adapter,
        ipp_eu
) -> None:
    result = nhits_adapter.fit(
        target=ipp_eu,
        exog=None,
        nhits_params=nhits_params,
        fit_params=fit_params
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    assert result.weight_path, "Путь к весам пуст"

    metrics = result.model_metrics.test_metrics
    for m in metrics:
        if m.type == "R2":
            assert m.value > 0, "Что-то не так с моделью, R2 меньше нуля, походу был шок в данных"
        elif m.type == "MAPE":
            assert m.value < 1
        elif m.type == "RMSE":
            assert m.value > 0.5, "Что-то не так с моделью, RMSE маленькое"

    # Проверка прогнозов
    train_predict_len = len(result.forecasts.train_predict.dates)
    test_predict_len = len(result.forecasts.test_predict.dates)
    future_predict_len = len(result.forecasts.forecast.dates)
    assert train_predict_len + test_predict_len == ipp_eu.shape[0]
    assert future_predict_len == fit_params.forecast_horizon - test_predict_len
