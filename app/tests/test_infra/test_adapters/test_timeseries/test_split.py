import pytest
from datetime import datetime
import pandas as pd

from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
from tests.conftest import sample_data_to_split, ts_splitter


def test_split_without_exog(sample_data_to_split, ts_splitter):
    target, _ = sample_data_to_split
    splitter = ts_splitter
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
    assert val_target.tolist() == [3, 4, 5]
    assert val_target.index.tolist() == pd.date_range("2020-01-04", "2020-01-06").tolist()

    # Проверка test
    assert exog_test is None
    assert test_target.tolist() == [6, 7, 8, 9]
    assert test_target.index.tolist() == pd.date_range("2020-01-07", "2020-01-10").tolist()


def test_split_with_exog(sample_data_to_split, ts_splitter):
    target, exog = sample_data_to_split
    splitter = ts_splitter
    train_boundary = datetime(2020, 1, 3)
    val_boundary = datetime(2020, 1, 6)

    # Вызов метода с exog
    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train
    assert exog_train["feature"].tolist() == [0, 1, 2]
    assert exog_train['feature 2'].tolist() == [10, 11, 12]
    assert train_target.tolist() == [0, 1, 2]

    # Проверка validation
    assert exog_val["feature"].tolist() == [3, 4, 5]
    assert exog_val['feature 2'].tolist() == [13, 14, 15]
    assert val_target.tolist() == [3, 4, 5]

    # Проверка test
    assert exog_test["feature"].tolist() == [6, 7, 8, 9]
    assert exog_test['feature 2'].tolist() == [16, 17, 18, 19]
    assert test_target.tolist() == [6, 7, 8, 9]


def test_split_boundaries_same(sample_data_to_split, ts_splitter):
    target, exog = sample_data_to_split
    splitter = ts_splitter
    train_boundary = datetime(2020, 1, 6)
    val_boundary = datetime(2020, 1, 6)  # Границы совпадают

    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train (до границы включительно)
    assert exog_train["feature"].tolist() == [0, 1, 2, 3, 4, 5]
    assert train_target.tolist() == [0, 1, 2, 3, 4, 5]

    # Проверка validation (пустой список)
    assert exog_val["feature"].tolist() == []
    assert val_target.tolist() == []

    # Проверка test (от границы включительно)
    assert exog_test["feature"].tolist() == [6, 7, 8, 9]
    assert test_target.tolist() == [6, 7, 8, 9]


def test_split_val_boundary_last(sample_data_to_split, ts_splitter):
    target, exog = sample_data_to_split
    splitter = ts_splitter
    train_boundary = datetime(2020, 1, 8)
    val_boundary = datetime(2020, 1, 10)  # Конец временного ряда

    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train
    assert exog_train["feature"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert train_target.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    # Проверка validation
    assert exog_val["feature"].tolist() == [8, 9]
    assert val_target.tolist() == [8, 9]

    # Проверка test (пустой список)
    assert exog_test["feature"].tolist() == []
    assert test_target.tolist() == []

def test_split_large_val_boundary(sample_data_to_split, ts_splitter):
    target, exog = sample_data_to_split
    splitter = ts_splitter
    train_boundary = datetime(2020, 1, 8)
    val_boundary = datetime(2021, 1, 10)  # Конец временного ряда

    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train
    assert exog_train["feature"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert train_target.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    # Проверка validation
    assert exog_val["feature"].tolist() == [8, 9]
    assert val_target.tolist() == [8, 9]

    # Проверка test (пустой список)
    assert exog_test["feature"].tolist() == []
    assert test_target.tolist() == []


def test_split_large_train_boundary(sample_data_to_split, ts_splitter):
    target, exog = sample_data_to_split
    splitter = ts_splitter
    train_boundary = datetime(2022, 1, 8)
    val_boundary = datetime(2021, 1, 10)  # Конец временного ряда

    exog_train, train_target, exog_val, val_target, exog_test, test_target = splitter.split(
        train_boundary, val_boundary, target, exog=exog
    )

    # Проверка train
    assert exog_train["feature"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert train_target.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Проверка validation
    assert exog_val["feature"].tolist() == []
    assert val_target.tolist() == []

    # Проверка test (пустой список)
    assert exog_test["feature"].tolist() == []
    assert test_target.tolist() == []