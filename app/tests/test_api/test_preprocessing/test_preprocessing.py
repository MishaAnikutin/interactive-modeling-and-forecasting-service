from itertools import permutations
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import boxcox

from src.core.application.preprocessing.preprocess_scheme import (
    DiffTransformation,
    LagTransformation,
    LogTransformation,
    PowTransformation,
    NormalizeTransformation,
    ExpSmoothTransformation,
    BoxCoxTransformation,
    FillMissingTransformation,
    MovingAverageTransformation,
    TransformationUnion
)
from src.core.domain import Timeseries, DataFrequency


def ipp_eu():
    df = pd.read_csv(
        "/Users/oleg/projects/interactive-modeling-and-forecasting-service/app/tests/data/month/ipc_eu.csv",
        sep=";"
    )
    df['date'] = pd.to_datetime(df['date'])
    target_ = pd.Series(data=df['value'].to_list(), index=df['date'].to_list(), name='ipp')
    return target_

target = ipp_eu()

ENDPOINT = "/api/v1/preprocessing/"

def serialise_ts(ts: Timeseries) -> dict:
    return {
        "name": ts.name,
        "values": ts.values,
        "dates": [date.strftime("%Y-%m-%d") for date in ts.dates],
        "data_frequency": ts.data_frequency,
    }

def delete_timestamp(ts: list[str]) -> list[str]:
    return [date.replace("T", " ") for date in ts]

def serialise_transformation(transformations: List[TransformationUnion]) -> dict:
    return [
        t.model_dump() for t in transformations
    ]


def make_request(ts: Timeseries, transformations: List[TransformationUnion]) -> Dict[str, Any]:
    """
    Полный JSON-запрос к эндпоинту.
    """
    return {
        "ts": serialise_ts(ts),
        "transformations": serialise_transformation(transformations),
    }


def to_series(ts_obj: dict[str, Any]) -> pd.Series:
    return pd.Series(
        data=ts_obj['values'],
        index=[date.replace("T00:00:00", "") for date in ts_obj['dates']],
        name=ts_obj['name']
    )

def from_series(series: pd.Series, freq: DataFrequency) -> Timeseries:
    return Timeseries(
        name=series.name,
        dates=series.index.tolist(),
        values=series.values.tolist(),
        data_frequency=freq
    )

def test_preprocessing_empty(client):
    ts = from_series(target, "ME")
    payload = make_request(ts, [])
    response = client.post(ENDPOINT, json=payload)

    data = response.json()

    assert response.status_code == 200, data
    assert data['values'] == ts.values
    assert data['name'] == ts.name
    assert data['data_frequency'] == ts.data_frequency
    assert delete_timestamp(data['dates']) == [str(date) for date in ts.dates]

def normalize(ts: pd.Series) -> pd.Series:
    return (ts - ts.mean()) / ts.std(ddof=0)

def my_boxcox(ts: pd.Series, lmbda: float) -> pd.Series:
    safe_ts = ts.copy()
    transformed = boxcox(safe_ts, lmbda=lmbda)
    return pd.Series(transformed, index=ts.index, name=ts.name)

@pytest.mark.parametrize(
    "transformations_list, expected",
    [
        (
            [
                DiffTransformation(diff_order=1),
            ],
            target.diff(1)
        ),
        (
            [
                DiffTransformation(diff_order=1),
                LagTransformation(lag_order=2),
            ],
            target.diff(1).shift(2)
        ),
        (
            [
                DiffTransformation(diff_order=3),
                LagTransformation(lag_order=2),
                DiffTransformation(diff_order=1),
            ],
            target.diff(3).shift(2).diff(1)
        ),
        (
            [
                DiffTransformation(diff_order=3),
                LagTransformation(lag_order=2),
                LogTransformation(),
                PowTransformation(pow_order=2),
                NormalizeTransformation(method="standard"),
                ExpSmoothTransformation(span=3),
                MovingAverageTransformation(window=4),
                FillMissingTransformation(method="last"),
                BoxCoxTransformation(param=0.5),
            ],
            my_boxcox(
                normalize(
                    (np.log(target.diff(3).shift(2)) ** 2)
                ).ewm(
                    span=3
                ).mean().rolling(
                    window=4,
                    min_periods=1
                ).mean().ffill(),
                0.5
            )
        )
    ]
)
def test_preprocessing_base(
        transformations_list,
        expected,
        client,
):
    ts = from_series(target, "ME")
    payload = make_request(ts, transformations_list)
    response = client.post(ENDPOINT, json=payload)

    data = response.json()

    assert response.status_code == 200, data

    expected_ts = from_series(expected, "ME")
    assert data['values'] == [None if pd.isna(val) else float(val) for val in expected_ts.values]
    assert data['name'] == expected_ts.name
    assert data['data_frequency'] == expected_ts.data_frequency
    assert delete_timestamp(data['dates']) == [str(date) for date in expected_ts.dates]


def test_unknown_transformation(client):
    """
    Неизвестный type → 422 (валидация Union Transformation).
    """
    transformations = [{"type": "foobar", "some": "value"}]
    ts = from_series(target, "ME")
    response = client.post(ENDPOINT, json={
        "ts": serialise_ts(ts),
        "transformations": transformations,
    })

    assert response.status_code == 422


permutated = list(permutations(
    [
        DiffTransformation(diff_order=3),
        LogTransformation(),
        PowTransformation(pow_order=2),
        NormalizeTransformation(method="standard"),
        ExpSmoothTransformation(span=3),
        MovingAverageTransformation(window=4),
        FillMissingTransformation(method="last"),
        BoxCoxTransformation(param=0.5),
    ]
))

@pytest.mark.parametrize(
    "transformations_list",
    [list(comb) for comb in permutated]
)
def test_permutations_of_transformations(client, transformations_list):
    # просто проверяем, что для любых перестановок преобразований ничего не сломается
    ts = from_series(target, "ME")
    payload = make_request(ts, transformations_list)
    response = client.post(ENDPOINT, json=payload)

    data = response.json()
    assert response.status_code == 200, data