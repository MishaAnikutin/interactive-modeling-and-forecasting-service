import pytest
import pandas as pd


@pytest.fixture
def metrics_factory():
    from src.infrastructure.adapters.metrics import MetricsFactory
    return MetricsFactory

@pytest.fixture
def sample_data():
    return {
        'y_true': pd.Series([3, -0.5, 2, 7]),
        'y_pred': pd.Series([2.5, 0.0, 2, 8])
    }

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