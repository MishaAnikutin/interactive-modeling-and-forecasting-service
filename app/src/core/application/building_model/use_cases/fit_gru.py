from .base import FitUC
from src.core.application.building_model.schemas.gru import GruFitRequest
from src.infrastructure.adapters.modeling.gru import GruAdapter


class FitGruUC(FitUC[GruFitRequest, GruAdapter]):
    pass
