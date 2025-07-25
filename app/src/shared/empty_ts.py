from src.core.domain import Timeseries, DataFrequency


def form_empty_ts(freq: DataFrequency) -> Timeseries:
    empty_ts = Timeseries(
        name="empty_ts",
        values=[],
        dates=[],
        data_frequency=freq,
    )
    return empty_ts