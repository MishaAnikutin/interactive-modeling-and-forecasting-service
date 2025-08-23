import pandas as pd
from fastapi import HTTPException

from src.core.domain import DataFrequency


def future_dates(
        last_dt: pd.Timestamp,
        data_frequency: DataFrequency,
        periods: int,
):
    if periods <= 0:
        return pd.DatetimeIndex([])
    freq_map: dict[DataFrequency, str] = {
        DataFrequency.year: "YE",
        DataFrequency.month: "ME",
        DataFrequency.quart: "QE",
        DataFrequency.day: "D",
    }
    try:
        freq_alias = freq_map[data_frequency]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемая частотность: {data_frequency}",
        )
    dr = pd.date_range(
        start=last_dt,
        periods=periods + 1,
        freq=freq_alias,
    )
    return dr[1:]
