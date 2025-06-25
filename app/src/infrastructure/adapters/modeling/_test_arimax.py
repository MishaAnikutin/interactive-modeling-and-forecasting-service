import pandas as pd

from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.modeling import ArimaxAdapter


def calculate_metrics():
    arimax_adapter = ArimaxAdapter(
        metric_factory=MetricsFactory(),
        ts_train_test_split=None
    )
    # 1. Подготовка тестовых данных
    y_train_true = pd.Series([10, 20, 30], index=pd.date_range("2023-01-01", periods=3))
    y_train_pred = pd.Series([12, 18, 33], index=pd.date_range("2023-01-01", periods=3))
    y_test_true = pd.Series([40, 50], index=pd.date_range("2023-01-04", periods=2))
    y_test_pred = pd.Series([38, 52], index=pd.date_range("2023-01-04", periods=2))

    # 3. Вызов тестируемого метода
    result = arimax_adapter._calculate_metrics(
        y_train_true=y_train_true,
        y_train_pred=y_train_pred,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred
    )

    print(result)

if __name__ == "__main__":
    calculate_metrics()