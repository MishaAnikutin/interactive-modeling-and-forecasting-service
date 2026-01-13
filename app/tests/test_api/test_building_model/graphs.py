import json
import matplotlib.pyplot as plt

import pandas as pd


def ts_to_pandas_series(ts):
    pandas_dates = pd.to_datetime(ts['dates'])
    series = pd.Series(
        data=ts['values'], index=pandas_dates, name=ts['name']
    )
    return series


def build_graph_v2():
    with open('../../data/fit_results/fit_request.json') as f:
        fit_request = json.load(f)

    # таргет
    target = fit_request['model_data']['dependent_variables']
    target_series = ts_to_pandas_series(target)

    with open('../../data/fit_results/fit_results.json') as f:
        fit_result = json.load(f)

    # лучший прогноз
    best_forecast = fit_result['best_forecast']
    best_forecast_series = ts_to_pandas_series(best_forecast)

    # оконные прогнозы
    forecasts = []
    for fcst in fit_result['forecasts']:
        fcst_series = ts_to_pandas_series(fcst)
        forecasts.append(fcst_series)

    # 1) построить график таргета
    plt.figure(figsize=(12, 6))
    plt.plot(target_series.index, target_series.values, label='Target', linewidth=2)

    # 2) построить график лучшего прогноза
    plt.plot(best_forecast_series.index, best_forecast_series.values,
             label='Best Forecast', linewidth=2, linestyle='--')

    # 3) построить график оконных прогнозов
    for i, forecast in enumerate(forecasts):
        plt.plot(forecast.index, forecast.values, alpha=0.5, label=f'Window Forecast {i + 1}')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Target vs Forecasts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    build_graph_v2()