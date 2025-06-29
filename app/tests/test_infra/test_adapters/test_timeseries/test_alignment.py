import pandas as pd
from datetime import datetime

from src.core.domain import Timeseries
from tests.conftest import ts_alignment


def test_alignment_base(ts_alignment):
    exog = [
        Timeseries(
            dates=[
                datetime(2017, 1, 1),
                datetime(2017, 2, 1),
                datetime(2017, 3, 1)
            ],
            values=[1.0, 2.0, 3.0],
            name='Ряд 1'
        ),
        Timeseries(
            dates=[
                datetime(2016, 12, 31),
                datetime(2017, 1, 30),
                datetime(2017, 2, 27)
            ],
            values=[4.0, 5.0, 6.0],
            name='Ряд 2'
        ),
    ]
    df = ts_alignment.compare(exog)

    df_expected = pd.DataFrame(
        index=['01.01.2017 00:00:00', '01.02.2017 00:00:00', '01.03.2017 00:00:00'],
        columns=['Ряд 1', 'Ряд 2']
    )
    df_expected['Ряд 1'] = [1.0, 2.0, 3.0]
    df_expected['Ряд 2'] = [4.0, 5.0, 6.0]

    pd.testing.assert_frame_equal(df, df_expected)


def test_alignment_biased(ts_alignment):
    exog = [
        Timeseries(
            dates=[
                datetime(2017, 2, 1),
                datetime(2017, 3, 1),
                datetime(2017, 4, 1)
            ],
            values=[1.0, 2.0, 3.0],
            name='Ряд 1'
        ),
        Timeseries(
            dates=[
                datetime(2016, 12, 31),
                datetime(2017, 1, 30),
                datetime(2017, 2, 27)
            ],
            values=[4.0, 5.0, 6.0],
            name='Ряд 2'
        ),
    ]
    df = ts_alignment.compare(exog)

    df_expected = pd.DataFrame(
        index=['01.02.2017 00:00:00', '01.03.2017 00:00:00'],
        columns=['Ряд 1', 'Ряд 2']
    )
    df_expected['Ряд 1'] = [1.0, 2.0]
    df_expected['Ряд 2'] = [5.0, 6.0]

    pd.testing.assert_frame_equal(df, df_expected)


def test_alignment_day_data(ts_alignment):
    exog = [
        Timeseries(
            dates=[
                datetime(2017, 1, 1),
                datetime(2017, 1, 2),
                datetime(2017, 1, 3)
            ],
            values=[1.0, 2.0, 3.0],
            name='Ряд 1'
        ),
        Timeseries(
            dates=[
                datetime(2017, 1, 1),
                datetime(2017, 1, 2),
                datetime(2017, 1, 3)
            ],
            values=[4.0, 5.0, 6.0],
            name='Ряд 2'
        ),
    ]
    df = ts_alignment.compare(exog)
    df_expected = pd.DataFrame(
        index=['01.01.2017 00:00:00', '02.01.2017 00:00:00', '03.01.2017 00:00:00'],
        columns=['Ряд 1', 'Ряд 2']
    )
    df_expected['Ряд 1'] = [1.0, 2.0, 3.0]
    df_expected['Ряд 2'] = [4.0, 5.0, 6.0]
    pd.testing.assert_frame_equal(df, df_expected)


def test_alignment_day_data_sec(ts_alignment):
    exog = [
        Timeseries(
            dates=[
                datetime(2017, 1, 1, 12, 30, 31),
                datetime(2017, 1, 2, 14, 41, 21),
                datetime(2017, 1, 3, 2, 15, 43)
            ],
            values=[1.0, 2.0, 3.0],
            name='Ряд 1'
        ),
        Timeseries(
            dates=[
                datetime(2017, 1, 1, 11, 30, 21),
                datetime(2017, 1, 2, 3, 3, 3),
                datetime(2017, 1, 3, 21, 40, 4)
            ],
            values=[4.0, 5.0, 6.0],
            name='Ряд 2'
        ),
    ]
    df = ts_alignment.compare(exog)
    df_expected = pd.DataFrame(
        index=['02.01.2017 00:00:00', '03.01.2017 00:00:00', '04.01.2017 00:00:00'],
        columns=['Ряд 1', 'Ряд 2']
    )
    df_expected['Ряд 1'] = [1.0, 2.0, 3.0]
    df_expected['Ряд 2'] = [4.0, 5.0, 6.0]
    pd.testing.assert_frame_equal(df, df_expected)

