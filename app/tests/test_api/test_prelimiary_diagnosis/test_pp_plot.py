from datetime import date
from scipy import stats
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import numpy as np

def gen_dates(num: int) -> list[date]:
    """Возвращает n дат типа «month ending», начиная с 31-01-2023."""
    dates_idx = pd.date_range(start="2023-01-31", periods=num, freq="ME")
    return [d.date().strftime("%Y-%m-%d") for d in dates_idx]

def gen_values(num: int):
    pageviews = np.random.normal(4, 4, size=num)
    return pageviews.tolist()

def process_ts():
    return {
        "dates": gen_dates(1000),
        "values": gen_values(1000),
        "name": "test_ts",
        "data_frequency": "ME"
    }


def test_qq(client):
    ts = process_ts()
    result = client.post(
        url='/api/v1/preliminary_diagnosis/data_representations/pp',
        json={
            "timeseries": ts,
            "distribution": "norm"
        }
    )
    data = result.json()
    assert result.status_code == 200, data

    theoretical_quantiles = np.array(data["theoretical_probs"])
    sample_sorted = np.array(data['empirical_probs'])

    plt.figure(figsize=(6, 6))
    plt.scatter(theoretical_quantiles, sample_sorted, s=12, alpha=0.8)
    plt.plot(
        [theoretical_quantiles.min(), theoretical_quantiles.max()],
        [theoretical_quantiles.min(), theoretical_quantiles.max()],
        color="red",
        lw=2,
    )
    plt.xlabel("Теоретические квантили")
    plt.ylabel("Выборочные квантили")
    plt.title("p–p plot duration, нормальное распредеелние")
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig("pp_plot.png")
