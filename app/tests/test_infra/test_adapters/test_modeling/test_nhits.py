import pytest
from tests.conftest import nhits_adapter, nhits_params_base, fit_params_base, ipp_eu

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_nhits_adapter_fit_without_exog(
        nhits_adapter,
        nhits_params_base,
        fit_params_base,
        ipp_eu
) -> None:
    result = nhits_adapter.fit(
        target=ipp_eu,
        exog=None,
        nhits_params=nhits_params_base,
        fit_params=fit_params_base
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    assert result.weight_path, "Путь к весам пуст"

    metrics = result.model_metrics.test_metrics
    for m in metrics:
        if m.type == "R2":
            assert m.value > 0, "Что-то не так с моделью, R2 меньше нуля, походу был шок в данных"
        elif m.type == "MAPE":
            assert m.value < 1
        elif m.type == "RMSE":
            assert m.value > 0.5, "Что-то не так с моделью, RMSE маленькое"