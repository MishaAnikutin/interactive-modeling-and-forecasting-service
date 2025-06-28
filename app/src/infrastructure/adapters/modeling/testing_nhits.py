from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.domain import FitParams, DataFrequency
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
from src.infrastructure.adapters.modeling.nhits import NhitsAdapter, NhitsParams


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_nhits_adapter_fit(tmp_path: Path) -> None:
    """
    Интеграционный тест: быстрая проверка обучения NHiTS-адаптера
    на синтетическом дневном ряде из 100 наблюдений.
    """

    # ---------- 1. Данные ----------------------------------------------------------
    n_obs = 100
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_obs, freq="D")
    values = np.sin(np.linspace(0, 10, n_obs)) + np.random.normal(scale=0.05, size=n_obs)
    assert len(dates) == len(values)
    target_series = pd.Series(values, index=dates, name="y")

    # ---------- 2. Гиперпараметры --------------------------------------------------
    nhits_params = NhitsParams(
        h=7,
        input_size=14,
        max_steps=5,
        early_stop_patience_steps=2,
        learning_rate=1e-2,
    )

    # ---------- 3. FitParams -------------------------------------------------------
    train_boundary = start_date + timedelta(days=69)   # первые 70 → train
    val_boundary = start_date + timedelta(days=84)     # 15 → val, 15 → test

    fit_params = FitParams(
        train_boundary=train_boundary,
        val_boundary=val_boundary,
        forecast_horizon=7,
        data_frequency=DataFrequency.day,
    )

    # ---------- 4. Инфраструктура --------------------------------------------------
    metric_factory = MetricsFactory()
    ts_splitter = TimeseriesTrainTestSplit()
    adapter = NhitsAdapter(
        metric_factory=metric_factory,
        ts_train_test_split=ts_splitter,
    )

    # ---------- 5. Обучение --------------------------------------------------------
    result = adapter.fit(
        target=target_series,
        exog=None,
        nhits_params=nhits_params,
        fit_params=fit_params,
    )

    # ---------- 6. Проверки --------------------------------------------------------
    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    print(result)

    #assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    #assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    #assert result.weight_path, "Путь к весам пуст"
    #assert Path(result.weight_path).exists(), "Файл весов не сохранён"