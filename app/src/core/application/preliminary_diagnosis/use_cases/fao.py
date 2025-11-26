from fastapi import HTTPException

from src.core.application.preliminary_diagnosis.errors.fao import InvalidFreq, SmallSizeError
from src.core.application.preliminary_diagnosis.schemas.fao import FaoRequest, FaoResult
from src.core.domain import DataFrequency
from src.infrastructure.adapters.preliminary_diagnosis.fao import FaoAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter, FrequencyDeterminer


class FaoUC:
    def __init__(
            self,
            fao_adapter: FaoAdapter,
            pandas_adapter: PandasTimeseriesAdapter,
            frequency_determiner: FrequencyDeterminer
    ):
        self._fao_adapter = fao_adapter
        self._pandas_adapter = pandas_adapter
        self._frequency_determiner = frequency_determiner

    def execute(self, request: FaoRequest) -> FaoResult:
        freq = self._frequency_determiner.determine(request.ts.dates)

        df = self._pandas_adapter.to_dataframe(request.ts)

        if freq != DataFrequency.month:
            if freq == DataFrequency.day:
                if df.shape[0] < 400:
                    raise HTTPException(status_code=400, detail=InvalidFreq().detail)
                df = df.resample("ME").mean(numeric_only=True)
            else:
                raise HTTPException(status_code=400, detail=InvalidFreq().detail)
        else:
            if df.shape[0] < 24:
                raise HTTPException(status_code=400, detail=SmallSizeError().detail)
        return self._fao_adapter.run(df)
