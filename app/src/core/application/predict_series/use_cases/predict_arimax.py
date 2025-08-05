import pandas as pd

from src.core.application.predict_series.schemas.schemas import PredictRequest, PredictResponse
from src.infrastructure.adapters.predicting.arimax import PredictArimaxAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class PredictArimaxUC:
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        predict_adapter: PredictArimaxAdapter
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._predict_adapter = predict_adapter

    def execute(self, request: PredictRequest) -> PredictResponse:
        self._ts_aligner.is_ts_freq_equal_to_expected(request.dependent_variables)

        # ---------------------------------- ОСТОРОЖНО!!! ГОВНОКОД!!!
        if request.explanatory_variables:
            df = self._ts_aligner.compare(
                timeseries_list=request.explanatory_variables,
                target=request.dependent_variables
            )

            target = df[request.dependent_variables.name]
            if type(target) == pd.DataFrame:
                target = target.iloc[:, 0]
            exog_df = df.drop(columns=[request.dependent_variables.name])
            if exog_df.empty:
                exog_df = None
        else:
            target = self._ts_adapter.to_series(request.dependent_variables)
            exog_df = None
        # ---------------------------------- КОНЕЦ ГОВНОКОДА

        in_sample, out_of_sample = self._predict_adapter.execute(
            model_weight=request.model_weight,
            steps=request.forecast_steps,
            target=target,
            exog_df=exog_df
        )

        freq = request.dependent_variables.data_frequency
        in_sample_predict = self._ts_adapter.from_series(in_sample, freq)
        out_of_sample_predict = self._ts_adapter.from_series(out_of_sample, freq)

        return PredictResponse(
            in_sample_predict=in_sample_predict,
            out_of_sample_predict=out_of_sample_predict
        )

