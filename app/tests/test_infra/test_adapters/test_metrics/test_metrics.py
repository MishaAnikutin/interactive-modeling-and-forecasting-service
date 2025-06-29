import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)


def test_factory_apply_basic_metrics(metrics_factory, sample_data):
    """Тест для базовых метрик (требуют только y_true и y_pred)"""
    metrics = ['MAPE', 'MAE', 'RMSE', 'MSE', 'R2']
    results = metrics_factory.apply(metrics=metrics, **sample_data)

    assert len(results) == len(metrics)

    # Проверяем корректность расчета для каждой метрики
    for metric in results:
        if metric.type == "MAPE":
            expected = mean_absolute_percentage_error(
                sample_data['y_true'], sample_data['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "MAE":
            expected = mean_absolute_error(
                sample_data['y_true'], sample_data['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "RMSE":
            expected = root_mean_squared_error(
                sample_data['y_true'], sample_data['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "MSE":
            expected = mean_squared_error(
                sample_data['y_true'], sample_data['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "R2":
            expected = r2_score(
                sample_data['y_true'], sample_data['y_pred']
            )
            assert metric.value == expected


def test_factory_apply_mase_metric(metrics_factory, mase_context):
    """Тест для метрики MASE с изолированным контекстом"""
    results = metrics_factory.apply(metrics=['MASE'], **mase_context)

    assert len(results) == 1
    metric = results[0]

    mae_i = mean_absolute_error(
        mase_context['y_true_i'], mase_context['y_pred_i']
    )
    mae_j = mean_absolute_error(
        mase_context['y_true_j'], mase_context['y_pred_j']
    )
    expected = mae_i / mae_j

    assert metric.type == "MASE"
    assert metric.value == pytest.approx(expected)


def test_factory_apply_adj_r2_metric(metrics_factory, sample_data, adj_r2_context):
    """Тест для скорректированного R-квадрата (требует дополнительные параметры)"""
    params = {**sample_data, **adj_r2_context}
    results = metrics_factory.apply(metrics=['AdjR2'], **params)

    assert len(results) == 1
    metric = results[0]

    r2 = r2_score(params['y_true'], params['y_pred'])
    n = params['row_count']
    p = params['feature_count']
    expected = 1 - (1 - r2) * (n - 1) / (n - p)

    assert metric.type == "Adj-R^2"
    assert metric.value == expected


def test_factory_apply_all_metrics(
        metrics_factory,
        sample_data,
        mase_context,
        adj_r2_context,
        all_metrics_config
):
    """Комплексный тест для всех метрик одновременно"""
    # Собираем все необходимые параметры
    context = {**sample_data, **mase_context, **adj_r2_context}
    results = metrics_factory.apply(metrics=all_metrics_config, **context)

    assert len(results) == len(all_metrics_config)

    # Проверяем что все метрики корректно рассчитаны
    for metric in results:
        if metric.type == "MAPE":
            expected = mean_absolute_percentage_error(
                context['y_true'], context['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "MAE":
            expected = mean_absolute_error(
                context['y_true'], context['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "RMSE":
            expected = root_mean_squared_error(
                context['y_true'], context['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "MSE":
            expected = mean_squared_error(
                context['y_true'], context['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "MASE":
            mae_i = mean_absolute_error(
                context['y_true_i'], context['y_pred_i']
            )
            mae_j = mean_absolute_error(
                context['y_true_j'], context['y_pred_j']
            )
            expected = mae_i / mae_j
            assert metric.value == expected
        elif metric.type == "R2":
            expected = r2_score(
                context['y_true'], context['y_pred']
            )
            assert metric.value == expected
        elif metric.type == "Adj-R^2":
            r2 = r2_score(context['y_true'], context['y_pred'])
            n = context['row_count']
            p = context['feature_count']
            expected = 1 - (1 - r2) * (n - 1) / (n - p)
            assert metric.value == expected