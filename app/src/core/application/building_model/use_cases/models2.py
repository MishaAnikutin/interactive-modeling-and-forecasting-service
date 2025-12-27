from src.infrastructure.adapters.modeling_2.nhits import NhitsAdapter2
from .base2 import FitUC2
from src.core.application.building_model.schemas import (
    ArimaxFitRequest,
    LstmFitRequest,
    NhitsFitRequest,
    GruFitRequest
)


class FitNhitsUC2(FitUC2[NhitsFitRequest, NhitsAdapter2]):
    pass

