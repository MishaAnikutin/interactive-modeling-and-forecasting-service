from .base import FitUC
from src.core.application.building_model.schemas import (
    ArimaxFitRequest,
    LstmFitRequest,
    NhitsFitRequest,
    GruFitRequest
)

from src.infrastructure.adapters.modeling import (
    ArimaxAdapter,
    GruAdapter,
    NhitsAdapter,
    LstmAdapter,
)


class FitArimaxUC(FitUC[ArimaxFitRequest, ArimaxAdapter]):
    pass


class FitGruUC(FitUC[GruFitRequest, GruAdapter]):
    pass


class FitNhitsUC(FitUC[NhitsFitRequest, NhitsAdapter]):
    pass


class FitLstmUC(FitUC[LstmFitRequest, LstmAdapter]):
    pass
