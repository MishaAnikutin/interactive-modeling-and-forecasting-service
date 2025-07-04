from datetime import datetime

import pytest
import pandas as pd


# ---------- Зависимости ----------
@pytest.fixture
def metrics_factory():
    from src.infrastructure.adapters.metrics import MetricsFactory
    return MetricsFactory

@pytest.fixture
def ts_splitter():
    from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
    return TimeseriesTrainTestSplit()

@pytest.fixture
def ts_alignment():
    from src.infrastructure.adapters.timeseries import TimeseriesAlignment
    return TimeseriesAlignment()

@pytest.fixture
def nhits_adapter(metrics_factory, ts_splitter):
    from src.infrastructure.adapters.modeling.nhits import NhitsAdapter
    adapter = NhitsAdapter(
        metric_factory=metrics_factory(),
        ts_train_test_split=ts_splitter,
    )
    return adapter

# ---------- Данные ----------
@pytest.fixture
def ipp_eu():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/ipc_eu.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = pd.Series(data=df['value'].to_list(), index=df['date'].to_list())
    return target

@pytest.fixture
def nhits_params_base():
    from src.infrastructure.adapters.modeling.nhits import NhitsParams

    return NhitsParams(
        max_steps=30,
        early_stop_patience_steps=3,
        val_check_steps=50,
        learning_rate=1e-3,
        scaler_type="robust",
    )


@pytest.fixture
def fit_params_base():
    from src.core.domain import FitParams, DataFrequency
    return FitParams(
        train_boundary=datetime(2016, 6, 30),
        val_boundary=datetime(2018, 5, 31),
        forecast_horizon=20,
        data_frequency=DataFrequency.month
    )


@pytest.fixture
def sample_data():
    return {
        'y_true': pd.Series([3, -0.5, 2, 7]),
        'y_pred': pd.Series([2.5, 0.0, 2, 8])
    }

@pytest.fixture
def sample_data_to_split():
    index = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    target = pd.Series(range(10), index=index, name="target")
    exog = pd.DataFrame({"feature": range(10), "feature 2": range(10, 20)}, index=index)
    return target, exog

@pytest.fixture
def mase_context():
    return {
        'y_true_i': pd.Series([3, -0.5, 2, 7]),
        'y_pred_i': pd.Series([2.5, 0.0, 2, 8]),
        'y_true_j': pd.Series([4, 1, 3]),
        'y_pred_j': pd.Series([3.5, 0.5, 2.5])
    }

@pytest.fixture
def adj_r2_context():
    return {
        'row_count': 10,
        'feature_count': 3
    }

@pytest.fixture
def all_metrics_config():
    return [
        'MAPE', 'MAE', 'RMSE', 'MSE',
        'MASE', 'R2', 'AdjR2'
    ]