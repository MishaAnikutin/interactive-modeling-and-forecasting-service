from pydantic import BaseModel, Field


class Histogram(BaseModel):
    centers: list[float] = Field(..., title="Центры столбцов")
    counts: list[float] = Field(..., title="Высоты столбцов")
    width: list[float] = Field(..., title="Ширина столбцов")


class EstimateHistogramParams(BaseModel):
    bins: int = Field(description='Число интервалов', default=32, gt=0, lt=1000)
    is_density: bool = Field(
        description="Если True то возвращает относительные частоты, иначе частоты",
        default=True
    )
