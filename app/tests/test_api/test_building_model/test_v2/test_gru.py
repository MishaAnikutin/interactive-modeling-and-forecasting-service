import io
import json
import zipfile
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

import pytest

from src.core.application.building_model.schemas.nhits_v2 import NhitsParams_V2
from src.core.domain import FitParams, Timeseries
from tests.test_api.test_building_model.validators import process_fit_params
from tests.test_api.utils import process_variable


def test_gru_baseline(
    client,
    balance,
    ipp_eu_ts,
):
    ipp_eu_ts_reduced = Timeseries(
        name=ipp_eu_ts.name,
        data_frequency=ipp_eu_ts.data_frequency,
        dates=ipp_eu_ts.dates[:30],
        values=ipp_eu_ts.values[:30],
    )
    min_date = max(balance.dates[0], ipp_eu_ts_reduced.dates[0])
    max_date = min(balance.dates[-1], ipp_eu_ts_reduced.dates[-1])
    min_date_index = balance.dates.index(min_date)
    max_date_index = balance.dates.index(max_date)
    dates = balance.dates[min_date_index:max_date_index]
    total_size = len(dates)
    aligned_balance = Timeseries(
        name="aligned_balance",
        data_frequency=balance.data_frequency,
        dates=dates,
        values=balance.values[min_date_index:max_date_index],
    )
    assert 29 == total_size
    assert len(aligned_balance.dates) == len(aligned_balance.values)
    # 2. Рассчет границ выборок
    train_size = total_size - 5 - 5
    train_end_idx = train_size - 1
    val_end_idx = train_end_idx + 5

    # 3. Установка границ дат
    train_boundary_date = dates[train_end_idx]
    val_boundary_date = dates[val_end_idx]

    fit_params = FitParams(
        train_boundary=train_boundary_date,
        val_boundary=val_boundary_date,
        forecast_horizon=12
    )

    nhits_params = NhitsParams_V2(
        input_size=8,
        output_size=3,
    )

    params = nhits_params.model_dump()
    params['loss'] = 'MAE'
    params['valid_loss'] = 'MAE'
    data = dict(
        model_data=dict(
            dependent_variables=process_variable(aligned_balance),
            explanatory_variables=[process_variable(ipp_eu_ts_reduced)]
        ),
        hyperparameters=params,
        fit_params=process_fit_params(fit_params),
    )

    result = client.post(
        url='/api/v2/building_model/nhits/fit',
        json=data
    )

    assert result.status_code == 200

    archive_bytes = result.content
    zip_buffer = io.BytesIO(archive_bytes)
    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
        file_list = zip_file.namelist()
        data_file = None
        for file_name in file_list:
            if 'json' in file_name:
                data_file = file_name
                break

        with zip_file.open(data_file) as f:
            data_dict = json.load(f)

    pandas_dates = pd.to_datetime(aligned_balance.dates)
    target_series = pd.Series(
        data=aligned_balance.values, index=pandas_dates, name=aligned_balance.name
    )

    best_forecast = data_dict['best_forecast']
    best_values = best_forecast['values']
    best_values = [round(i, 3) if -100<i<100 else -40 for i in best_values]
    print(best_values)
    best_forecast_series = pd.Series(
        data=best_values, index=pd.to_datetime(best_forecast['dates']), name=best_forecast['name'])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(target_series, color='black', label='Target', linewidth=2)
    ax.plot(best_forecast_series, color='red', label='Best Forecast', linewidth=1.5)

    # Добавляем области разных датасетов
    train_end = pd.Timestamp(train_boundary_date)
    ax.axvline(x=train_end, color='green', linestyle='--',
               linewidth=1.5, label='Train/Val boundary')

    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax],
                     target_series.index[0], train_end,
                     alpha=0.1, color='green', label='Train')

    val_end = pd.Timestamp(val_boundary_date)
    ax.axvline(x=val_end, color='orange', linestyle='--',
               linewidth=1.5, label='Val/Test boundary')

    ax.fill_betweenx([ymin, ymax],
                     train_end, val_end,
                     alpha=0.1, color='orange', label='Validation')

    forecast_start = target_series.index[-1]
    ax.axvline(x=forecast_start, color='purple', linestyle=':',
               linewidth=2, label='Forecast start')


    # Настройки графика
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'NHITS Model - Predictions Visualization', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.show()
