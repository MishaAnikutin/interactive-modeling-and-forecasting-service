from dishka import make_container

from .core_provider import CoreProvider
from .infra_provider import InfraProvider
from .logs_provider import LogsProvider

container = make_container(CoreProvider(), InfraProvider(), LogsProvider())


__all__ = ("container",)
