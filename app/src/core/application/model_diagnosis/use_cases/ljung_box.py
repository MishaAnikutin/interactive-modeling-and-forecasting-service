from statsmodels.stats.diagnostic import acorr_ljungbox

from src.core.application.model_diagnosis.schemas.ljung_box import LjungBoxResult, LjungBoxRequest
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class LjungBoxUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: LjungBoxRequest) -> LjungBoxResult:
        target, _ = self._aligned_data(request.data.target, exog=None)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts)
        )
        result_df = acorr_ljungbox(
            residuals,
            lags=request.lags,
            boxpierce=True,
            model_df=request.model_df,
            period=request.period,
            auto_lag=request.auto_lag,
        )

        print(result_df)

        return LjungBoxResult(
            lb_stat=result_df.lb_stat.to_list(),
            lb_pvalue=result_df.lb_pvalue.to_list(),
            bp_stat=result_df.bp_stat.to_list(),
            bp_pvalue=result_df.bp_pvalue.to_list(),
        )
