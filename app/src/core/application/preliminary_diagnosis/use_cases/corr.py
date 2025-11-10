import pandas as pd

from src.core.application.preliminary_diagnosis.schemas.corr import (CorrelationAnalysisResponse,
                                                                     CorrelationAnalysisRequest)
from src.core.domain.correlation.correlation import CorrelationMatrix
from src.infrastructure.adapters.timeseries import TimeseriesAlignment
from src.infrastructure.interactors.correlation import CorrelationInteractor


class CorrelationMatrixUC:
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            corr_adapter: CorrelationInteractor
    ):
        self._ts_aligner = ts_aligner
        self._corr_adapter = corr_adapter

    def execute(self, request: CorrelationAnalysisRequest) -> CorrelationAnalysisResponse:
        # FIXME: aligner дебильно написан, почему то зависит от контекста обучения моделей
        dataframe: pd.DataFrame = self._ts_aligner.compare(
            timeseries_list=request.variables[1:],
            target=request.variables[0]
        )

        correlation_matrix: CorrelationMatrix = self._corr_adapter.calculate(
            dataframe=dataframe,
            method=request.method
        )

        return CorrelationAnalysisResponse(correlation_matrix=correlation_matrix)
