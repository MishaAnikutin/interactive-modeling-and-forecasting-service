from pydantic import Field, model_validator
from enum import Enum

from src.core.application.preliminary_diagnosis.errors.kpss import InvalidLagsError
from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams


class RegressionEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'

class NlagsEnum(str, Enum):
    auto = 'auto'
    legacy = 'legacy'

class KpssParams(StatTestParams):
    regression: RegressionEnum = Field(
        default=RegressionEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест",
        description="Компонента тренда, которую следует включить в тест",
    )
    nlags: NlagsEnum | int = Field(
        default=NlagsEnum.auto,
        title="Число лагов",
        description=(
            "Параметр указывает количество лагов, которое будет использовано: "
            "1) Если установлено значение `auto` (по умолчанию), количество лагов рассчитывается автоматически "
            "с использованием метода, зависящего от данных, предложенного Хобайном и др. (1998). "
            "Дополнительные ссылки: Эндрюс (1991), Ньюи и Вест (1994), Шверт (1989). "
            "2) Если установлено значение `legacy`, используется формула `int(12 * (n / 100)**(1 / 4))`, "
            "описанная в Шверте (1989), где `n` — количество наблюдений. "
            "3) Если число, то должно быть меньше числа наблюдений и больше или равно 0"
        )
    )

    @model_validator(mode='after')
    def validate_nlags(self):
        if type(self.nlags) != NlagsEnum:
            if self.nlags >= len(self.ts.values) or self.nlags < 0:
                raise ValueError(InvalidLagsError().detail)
        return self
