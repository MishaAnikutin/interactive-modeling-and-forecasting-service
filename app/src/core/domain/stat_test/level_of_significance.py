from typing import TypeVar, Literal

SignificanceLevel = TypeVar('SignificanceLevel', bound=Literal[0.1, 0.05, 0.01])
