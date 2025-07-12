import pytest
import pandas as pd
from src.core.domain import Timeseries, DataFrequency


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
def freq_determiner():
    from src.infrastructure.adapters.timeseries import FrequencyDeterminer
    return FrequencyDeterminer()

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


@pytest.fixture
def ts_adapter():
    from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
    return PandasTimeseriesAdapter

@pytest.fixture
def preprocess_factory():
    from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory
    return PreprocessFactory

@pytest.fixture
def app():
    from dishka.integrations.fastapi import setup_dishka
    from main import create_fastapi_app
    from src.api import container
    app = create_fastapi_app()
    setup_dishka(container, app)
    return app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)

# ---------- Данные ----------
@pytest.fixture
def ipp_eu():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/month/ipc_eu.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = pd.Series(data=df['value'].to_list(), index=df['date'].to_list())
    return target

@pytest.fixture
def ipp_eu_ts():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/month/ipc_eu.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value'].to_list(),
        dates=df['date'].to_list(),
        name="ipp_eu",
        data_frequency=DataFrequency.month,
    )
    assert len(target.values) == len(target.dates)
    return target


@pytest.fixture
def u_total():
    return u_total_ts()

def u_total_ts():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/year/u_total.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value1'].to_list(),
        dates=df['date'].to_list(),
        name="u_total",
        data_frequency=DataFrequency.year,
    )
    assert len(target.values) == len(target.dates)
    return target


@pytest.fixture
def u_women():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/year/u_women.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value1'].to_list(),
        dates=df['date'].to_list(),
        name="u_women",
        data_frequency=DataFrequency.year,
    )
    assert len(target.values) == len(target.dates)
    return target


@pytest.fixture
def u_men():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/year/u_men.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value1'].to_list(),
        dates=df['date'].to_list(),
        name="u_men",
        data_frequency=DataFrequency.year,
    )
    assert len(target.values) == len(target.dates)
    return target


def balance_ts():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/month/balance.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value1'].to_list(),
        dates=df['date'].to_list(),
        name="balance",
        data_frequency=DataFrequency.month,
    )
    assert len(target.values) == len(target.dates)
    return target

@pytest.fixture
def balance():
    return balance_ts()

@pytest.fixture
def labour():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/quarter/labour.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value1'].to_list(),
        dates=df['date'].to_list(),
        name="labour",
        data_frequency=DataFrequency.quart,
    )
    assert len(target.values) == len(target.dates)
    return target

def ca_ts():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/quarter/ca.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target = Timeseries(
        values=df['value1'].to_list(),
        dates=df['date'].to_list(),
        name="ca",
        data_frequency=DataFrequency.quart,
    )
    assert len(target.values) == len(target.dates)
    return target

@pytest.fixture
def ca():
    return ca_ts()

# Параметры для тестов

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