from dishka import Provider, Scope, provide
from logging import Logger

from logs import logger as _root_logger


class LoggerProvider(Provider):
    scope = Scope.REQUEST

    app_logger: Logger = provide(source=lambda: _root_logger.getChild("app")    )