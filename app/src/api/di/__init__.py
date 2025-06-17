from dishka import make_container

from .core_provider import CoreProvider
from .infra_provider import InfraProvider


container = make_container(CoreProvider(), InfraProvider())


__all__ = ("container",)
