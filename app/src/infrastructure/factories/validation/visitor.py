from typing import Optional

import pandas as pd
from functools import wraps

from src.core.domain.validation import ValidationType, ValidationStrategyI, ValidationIssue


class ValidationVisitor:
    _registry: dict[ValidationType, type[ValidationStrategyI]] = {}

    @classmethod
    def register(cls, name: ValidationType):
        @wraps(cls.register)
        def wrapper(class_: type[ValidationStrategyI]):
            cls._registry[name] = class_
            return class_

        return wrapper

    @classmethod
    def available_validators(cls) -> list[ValidationStrategyI]:
        return [cls._registry[key]() for key in cls._registry.keys()]

    @classmethod
    def check(cls, ts: pd.Series) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = list()

        for validator in cls.available_validators():
            validation_issue: Optional[ValidationIssue] = validator.check(ts)

            if validation_issue is not None:
                issues.append(validation_issue)

        return issues
