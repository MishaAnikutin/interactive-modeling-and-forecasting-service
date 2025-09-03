from src.core.application.preliminary_diagnosis.schemas.acf_and_pacf import AcfAndPacfRequest, AcfPacfResult

from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from statsmodels.tsa.stattools import acf, pacf


class AcfAndPacfUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: AcfAndPacfRequest) -> AcfPacfResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)

        acf_values, acf_confint = acf(ts, nlags=request.nlags, alpha=request.alpha)
        pacf_values, pacf_confint = pacf(ts, method=request.pacf_method.value, nlags=request.nlags, alpha=request.alpha)

        print(f'{[float(el) for el in acf_values] = }')

        return AcfPacfResult(
            acf_values=[float(el) for el in acf_values],
            acf_confint=list(acf_confint),
            pacf_values=[float(el) for el in pacf_values],
            pacf_confint=list(pacf_confint)
        )
