from src.infrastructure.adapters.modeling import ArimaxAdapter, NhitsAdapter, LstmAdapter
from .base import FitUC
from src.core.application.building_model.schemas.gru import GruFitRequest
from src.infrastructure.adapters.modeling.gru import GruAdapter
from ..schemas.arimax import ArimaxFitRequest
from ..schemas.lstm import LstmFitRequest
from ..schemas.nhits import NhitsFitRequest


class FitArimaxUC(FitUC[ArimaxFitRequest, ArimaxAdapter]):
    pass

class FitGruUC(FitUC[GruFitRequest, GruAdapter]):
    pass

class FitNhitsUC(FitUC[NhitsFitRequest, NhitsAdapter]):
    pass

class FitLstmUC(FitUC[LstmFitRequest, LstmAdapter]):
    pass
