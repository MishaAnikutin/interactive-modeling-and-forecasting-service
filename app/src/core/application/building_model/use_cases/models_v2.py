from src.infrastructure.adapters.modeling_2.gru import GruAdapter_V2
from src.infrastructure.adapters.modeling_2.lstm import LstmAdapter_V2
from src.infrastructure.adapters.modeling_2.nhits import NhitsAdapter_V2
from .base_v2 import FitUC_V2
from src.core.application.building_model.schemas import (
    LstmFitRequest_V2,
    NhitsFitRequest_V2,
    GruFitRequest_V2
)


class FitNhitsUC_V2(FitUC_V2[NhitsFitRequest_V2, NhitsAdapter_V2]):
    pass

class FitLstmUC_V2(FitUC_V2[LstmFitRequest_V2, LstmAdapter_V2]):
    pass

class FitGruUC_V2(FitUC_V2[GruFitRequest_V2, GruAdapter_V2]):
    pass

