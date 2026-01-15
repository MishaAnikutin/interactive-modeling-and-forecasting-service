from pydantic import model_validator, BaseModel
from src.core.domain.stat_test import Conclusion
from src.shared.utils import validate_float_param


class DmTestResult(BaseModel):
    statistic: float
    p_value: float
    conclusion: Conclusion

    @model_validator(mode="after")
    def validate_value(self):
        self.p_value = validate_float_param(self.p_value)
        self.statistic = validate_float_param(self.statistic)
        return self
