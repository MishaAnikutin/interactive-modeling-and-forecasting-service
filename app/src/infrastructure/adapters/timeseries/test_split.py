import pytest
from datetime import datetime
import pandas as pd

from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


@pytest.fixture
def sample_data():
    index = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    target = pd.Series(range(10), index=index, name="target")
    exog = pd.DataFrame({"feature": range(10)}, index=index)
    return target, exog


def test_split_without_exog(sample_data):
    target, _ = sample_data
    splitter = TimeseriesTrainTestSplit()
    train_boundary = datetime(2020, 1, 3)
    val_boundary = datetime(2020, 1, 6)

    # Вызов метода без exog
    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=None
    )

    # Проверка train
    assert exog_train is None
    assert train_target.tolist() == [0, 1, 2]
    assert train_target.index.tolist() == pd.date_range("2020-01-01", "2020-01-03").tolist()

    # Проверка validation
    assert exog_val is None
    assert val_target.tolist() == [2, 3, 4, 5]
    assert val_target.index.tolist() == pd.date_range("2020-01-03", "2020-01-06").tolist()

    # Проверка test
    assert exog_test is None
    assert test_target.tolist() == [5, 6, 7, 8, 9]
    assert test_target.index.tolist() == pd.date_range("2020-01-06", "2020-01-10").tolist()


def test_split_with_exog(sample_data):
    target, exog = sample_data
    splitter = TimeseriesTrainTestSplit()
    train_boundary = datetime(2020, 1, 3)
    val_boundary = datetime(2020, 1, 6)

    # Вызов метода с exog
    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train
    assert exog_train["feature"].tolist() == [0, 1, 2]
    assert train_target.tolist() == [0, 1, 2]

    # Проверка validation
    assert exog_val["feature"].tolist() == [2, 3, 4, 5]
    assert val_target.tolist() == [2, 3, 4, 5]

    # Проверка test
    assert exog_test["feature"].tolist() == [5, 6, 7, 8, 9]
    assert test_target.tolist() == [5, 6, 7, 8, 9]


def test_split_boundaries_same(sample_data):
    target, exog = sample_data
    splitter = TimeseriesTrainTestSplit()
    train_boundary = datetime(2020, 1, 6)
    val_boundary = datetime(2020, 1, 6)  # Границы совпадают

    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train (до границы включительно)
    assert exog_train["feature"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert train_target.tolist() == [0, 1, 2, 3, 4, 5, 6]

    # Проверка validation (одна точка - граница)
    assert exog_val["feature"].tolist() == [6]
    assert val_target.tolist() == [6]

    # Проверка test (от границы включительно)
    assert exog_test["feature"].tolist() == [6, 7, 8, 9]
    assert test_target.tolist() == [6, 7, 8, 9]


def test_split_val_boundary_last(sample_data):
    target, exog = sample_data
    splitter = TimeseriesTrainTestSplit()
    train_boundary = datetime(2020, 1, 8)
    val_boundary = datetime(2020, 1, 10)  # Конец временного ряда

    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train
    assert exog_train["feature"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert train_target.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    # Проверка validation
    assert exog_val["feature"].tolist() == [7, 8, 9]
    assert val_target.tolist() == [7, 8, 9]

    # Проверка test (одна точка - граница)
    assert exog_test["feature"].tolist() == [9]
    assert test_target.tolist() == [9]