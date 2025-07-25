from src.core.domain import Forecasts, Timeseries
from src.shared.empty_ts import form_empty_ts
from src.core.application.model_diagnosis.errors.common import NotEqualFreqError, NotEqualLenError, NotEqualDatesError


def get_full_predict(ts: Timeseries, forecasts: Forecasts) -> Timeseries:
    target_freq = ts.data_frequency
    validation_predict = forecasts.validation_predict if forecasts.validation_predict else form_empty_ts(
        target_freq)
    test_predict = forecasts.test_predict if forecasts.test_predict else form_empty_ts(target_freq)

    full_values = forecasts.train_predict.values + validation_predict.values + test_predict.values
    full_dates = forecasts.train_predict.dates + validation_predict.dates + test_predict.dates
    full_predict = Timeseries(
        values=full_values,
        dates=full_dates,
        data_frequency=target_freq,
        name="Прогноз " + ts.name
    )

    if validation_predict.data_frequency != target_freq or test_predict.data_frequency != target_freq:
        raise ValueError(NotEqualFreqError().detail)
    if len(full_predict.dates) != len(ts.dates):
        raise ValueError(NotEqualLenError().detail)
    if len(full_predict.values) != len(ts.values):
        raise ValueError(NotEqualLenError().detail)
    if full_predict.dates != ts.dates:
        raise ValueError(NotEqualDatesError().detail)
    return full_predict