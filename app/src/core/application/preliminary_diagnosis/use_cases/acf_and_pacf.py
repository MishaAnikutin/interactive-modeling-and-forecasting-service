from src.core.application.preliminary_diagnosis.schemas.acf_and_pacf import AcfAndPacfRequest, AcfPacfResult

from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import statsmodels.tsa.stattools as tsstats


class AcfAndPacfUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: AcfAndPacfRequest) -> AcfPacfResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)

        acf_values, confint, qstat, pvals = tsstats.acf(
            ts,
            nlags=request.nlags,
            alpha=request.alpha,
            bartlett_confint=False,
            qstat=True
        )
        acf_confint = [(a - c, b - c) for (a, b), c in zip(confint, acf_values)]
        acf_confint = acf_confint[1:]
        acf_values = acf_values[1:]

        pacf_values, confint = tsstats.pacf(
            ts,
            method=request.pacf_method.value,
            nlags=request.nlags,
            alpha=request.alpha
        )
        pacf_confint = [(a - c, b - c) for (a, b), c in zip(confint, pacf_values)]
        pacf_confint = pacf_confint[1:]
        pacf_values = pacf_values[1:]

        return AcfPacfResult(
            acf_values=[float(el) for el in acf_values],
            acf_confint=list(acf_confint),
            pacf_values=[float(el) for el in pacf_values],
            pacf_confint=list(pacf_confint)
        )
