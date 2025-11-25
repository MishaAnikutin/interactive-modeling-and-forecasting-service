from src.core.application.preliminary_diagnosis.schemas.kim_andrews import KimAndrewsRequest, KimAndrewsResult
from src.core.application.preprocessing.preprocess_scheme import LagTransformation
from src.infrastructure.adapters.preliminary_diagnosis.kim_andrews import KimAndrewsAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.infrastructure.factories.preprocessing import PreprocessFactory


class KimAndrewsUC:
    def __init__(
            self,
            kim_andrews_adapter: KimAndrewsAdapter,
    ):
        self._kim_andrews_adapter = kim_andrews_adapter

    def execute(self, request: KimAndrewsRequest) -> KimAndrewsResult:
        result = self._kim_andrews_adapter.run(request.ts, request.n, request.m, request.shift, request.trend, request.const)
        return result