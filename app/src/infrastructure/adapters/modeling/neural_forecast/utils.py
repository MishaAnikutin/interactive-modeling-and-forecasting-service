import pandas as pd
from neuralforecast import NeuralForecast

from src.core.domain import DataFrequency
from src.shared.future_dates import future_dates
from src.shared.to_panel import to_panel


def form_train_df(
    exog: pd.DataFrame | None,
    train_target: pd.Series,
    val_target: pd.Series,
    exog_train: pd.DataFrame | None,
    exog_val: pd.DataFrame | None,
):
    val_size = val_target.shape[0]
    if exog is not None:
        train_df = to_panel(
            target=pd.concat([train_target, val_target]) if val_size != 0 else train_target,
            exog=pd.concat([exog_train, exog_val]) if exog_val.shape[0] != 0 else exog_train,
        )
    else:
        train_df = to_panel(
            target=pd.concat([train_target, val_target]) if val_size != 0 else train_target
        )
    return train_df


def form_future_df(
        future_size: int,
        test_target: pd.Series,
        freq: DataFrequency,
):
    last_known_dt = test_target.index.max()
    futr_index = future_dates(
        last_dt=last_known_dt,
        data_frequency=freq,
        periods=future_size,
    )
    futr_index_expanded = (
        pd.concat([test_target, pd.Series(index=futr_index)]).index
        if test_target.shape[0] != 0 else futr_index
    )

    return pd.DataFrame(
        {
            "unique_id": 'ts',
            "ds": futr_index_expanded,
        }
    )

def full_train_predict(
    model_name: str,
    nf: NeuralForecast,
    train_df: pd.DataFrame,
    val_size: int,
):
    fcst_insample_df = nf.predict_insample()
    fcst_train = (
        fcst_insample_df.loc[fcst_insample_df['ds'].isin(train_df['ds'])]
        .drop_duplicates('ds', keep='last')
        .set_index('ds')[model_name]
    )
    if val_size > 0:
        train_predict = fcst_train.iloc[:-val_size]
        validation_predict = fcst_train.iloc[-val_size:]
    else:
        train_predict = fcst_train.copy()
        validation_predict = pd.Series()
    return train_predict, validation_predict

def full_predict(
    model_name: str,
    nf: NeuralForecast,
    future_df: pd.DataFrame,
    test_size: int,
):
    all_forecasts = nf.predict(futr_df=future_df)[model_name]
    all_forecasts.index = future_df['ds']
    if test_size > 0:
        fcst_test = all_forecasts.iloc[:test_size].copy()
        fcst_future = all_forecasts.iloc[test_size:].copy()
    else:
        fcst_test = pd.Series()
        fcst_future = all_forecasts.copy()
    return fcst_test, fcst_future
