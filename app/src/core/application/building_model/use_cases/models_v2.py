from src.infrastructure.adapters.modeling_2.nhits import NhitsAdapter_V2
from .base_v2 import FitUC_V2
from src.core.application.building_model.schemas import (
    ArimaxFitRequest,
    LstmFitRequest,
    NhitsFitRequest_V2,
    GruFitRequest
)


class FitNhitsUC_V2(FitUC_V2[NhitsFitRequest_V2, NhitsAdapter_V2]):
    pass

