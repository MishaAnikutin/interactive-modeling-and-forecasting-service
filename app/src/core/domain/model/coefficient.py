from pydantic import BaseModel


class Coefficient(BaseModel):
    name: str
    value: float
    p_value: float
