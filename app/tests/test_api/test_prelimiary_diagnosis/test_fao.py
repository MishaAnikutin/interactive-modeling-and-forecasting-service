from datetime import date

import pytest
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.infrastructure.adapters.preliminary_diagnosis.fao import proc_res


def gen_dates(num: int) -> list[date]:
    """Возвращает n дат типа «month ending», начиная с 31-01-2023."""
    dates_idx = pd.date_range(start="2023-01-31", periods=num, freq="ME")
    return [d.date().strftime("%Y-%m-%d") for d in dates_idx]

def gen_dates_day(num: int) -> list[date]:
    dates_idx = pd.date_range(start="2023-01-31", periods=num, freq="D")
    return [d.date().strftime("%Y-%m-%d") for d in dates_idx]

def gen_values(num: int):
    pageviews = np.random.normal(4, 4, size=num)
    return pageviews.tolist()

def process_ts(num = 1000):
    return {
        "dates": gen_dates(num),
        "values": gen_values(num),
        "name": "test_ts",
        "data_frequency": "ME"
    }

def process_ts_day(num = 1000):
    return {
        "dates": gen_dates_day(num),
        "values": gen_values(num),
        "name": "test_ts",
        "data_frequency": "ME"
    }

def fao_plot(
    data: pd.DataFrame,
    ts: pd.Series,
    title: str,
    file: str,
) -> None:
    df = data.copy()
    df["UN_colors"] = df["UN_result"].apply(lambda x: proc_res(x))
    df["PC_colors"] = df["PC_result"].apply(lambda x: proc_res(x))

    cmap = ListedColormap(["white", "yellow", "red"])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(ts)
    ax.pcolor(
        ts.index,
        ax.get_ylim(),
        df[["UN_colors", "PC_colors"]].T,
        cmap=cmap,
        alpha=0.2,
        linewidth=0,
        antialiased=True,
        shading="nearest",
    )
    ax.text(
        0.05,
        0.05,
        "UN weights",
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="red",
        fontsize=15,
    )
    ax.text(
        0.05,
        0.95,
        "Eigen-values weights",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="green",
        fontsize=15,
    )

    fig.suptitle(title)
    plt.savefig(file)


def test_fao(client):
    ts = process_ts(130)
    result = client.post(
        url='/api/v1/preliminary_diagnosis/data_representations/fao',
        json={"ts": ts}
    )
    data = result.json()
    assert result.status_code == 200, data

    data = pd.DataFrame(data)

    ts = pd.Series(ts['values'], index=pd.to_datetime(ts['dates'], format='%Y-%m-%d'), name=ts['name'])

    fao_plot(data, ts, title='Fao plot', file='fao_plot.png')

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("num", list(range(10, 40, 3)) + [100, 1000])
def test_fao_short(num, client):
    ts = process_ts(num)
    result = client.post(
        url='/api/v1/preliminary_diagnosis/data_representations/fao',
        json={"ts": ts}
    )
    data = result.json()
    if num >= 24:
        assert result.status_code == 200, data
        data = pd.DataFrame(data)
        ts = pd.Series(ts['values'], index=pd.to_datetime(ts['dates'], format='%Y-%m-%d'), name=ts['name'])
        fao_plot(data, ts, title='Fao plot', file='fao_plot.png')
    else:
        assert result.status_code == 400, num

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("num", list(range(200, 3000, 100)))
def test_fao_day_freq(num, client):
    ts = process_ts_day(num)
    result = client.post(
        url='/api/v1/preliminary_diagnosis/data_representations/fao',
        json={"ts": ts}
    )
    data = result.json()
    if num >= 400:
        assert result.status_code == 200, data
        data = pd.DataFrame(data)
        ts = pd.Series(ts['values'], index=pd.to_datetime(ts['dates'], format='%Y-%m-%d'), name=ts['name'])
        ts = ts.resample("ME").mean(numeric_only=True)
        fao_plot(data, ts, title='Fao plot day freq', file='fao_plot_day.png')
    else:
        assert result.status_code == 400, num
