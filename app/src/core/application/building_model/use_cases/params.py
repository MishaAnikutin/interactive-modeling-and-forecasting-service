from src.core.application.building_model.schemas.arimax import ArimaxParams
from src.core.application.building_model.schemas.nhits import NhitsParams


class ArimaxParamsUC:
    def execute(self) -> ArimaxParams:
        return ArimaxParams()

class NhitsParamsUC:
    def execute(self) -> NhitsParams:
        return NhitsParams()