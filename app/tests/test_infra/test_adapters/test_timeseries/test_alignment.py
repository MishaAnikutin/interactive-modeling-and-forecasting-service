import pandas as pd
from datetime import datetime

import pytest
from fastapi import HTTPException

from src.core.domain import Timeseries, DataFrequency
from tests.conftest import ts_alignment, u_total, u_women, u_men, ipp_eu_ts


def test_alignment_base(ts_alignment, u_total, u_women, u_men):
    exog = [u_men, u_women]
    df = ts_alignment.compare(timeseries_list=exog, target=u_total)
    assert df.shape == (len(u_total.dates), 3)
    assert df.index.to_list() == u_total.dates
    assert df.columns.to_list() == [u_total.name, u_men.name, u_women.name]
    assert df.index.to_list() == u_men.dates
    assert df.index.to_list() == u_women.dates


def test_alignment_different_frequencies_exog(ts_alignment, u_total, u_men, ipp_eu_ts):
    exog = [u_men, ipp_eu_ts]
    with pytest.raises(HTTPException) as exc:
        ts_alignment.compare(timeseries_list=exog, target=u_total)
    assert exc.value.status_code == 400
    assert (
        "Частотность экзогенной переменной ipp_eu не соответствует частотности целевой переменной"
        in exc.value.detail
    )

def test_alignment_broken_frequencies(ts_alignment, u_total, u_men):
    u_men_broken = Timeseries(
        name=u_men.name,
        dates=u_men.dates,
        values=u_men.values,
        data_frequency=DataFrequency.month # поменял тут на месячный
    )
    u_total_broken = Timeseries(
        name=u_total.name,
        dates=u_total.dates,
        values=u_total.values,
        data_frequency=DataFrequency.month
    )
    # случай когда экзогенная переменная сломана
    exog = [u_men_broken]
    with pytest.raises(HTTPException) as exc:
        ts_alignment.compare(timeseries_list=exog, target=u_total)
    assert exc.value.status_code == 400
    assert (
        "Не соответствует полученных тип ряда и заявленный"
        in exc.value.detail
    )
    # случай когда таргет сломан
    exog = [u_men]
    with pytest.raises(HTTPException) as exc:
        ts_alignment.compare(timeseries_list=exog, target=u_total_broken)
    assert exc.value.status_code == 400
    assert (
            "Не соответствует полученных тип ряда и заявленный"
            in exc.value.detail
    )

def test_alignment_different_dates(ts_alignment, u_total, u_men, u_women):
    u_total_first_20 = Timeseries(
        name=u_total.name,
        data_frequency=u_total.data_frequency,
        dates=u_total.dates[:20],
        values=u_total.values[:20],
    )
    u_men_last_20 = Timeseries(
        name=u_men.name,
        data_frequency=u_men.data_frequency,
        dates=u_men.dates[-20:],
        values=u_men.values[-20:],
    )
    exog = [u_men_last_20, u_women]
    df = ts_alignment.compare(timeseries_list=exog, target=u_total_first_20)
    assert df.index.to_list() == u_women.dates[12:20]
