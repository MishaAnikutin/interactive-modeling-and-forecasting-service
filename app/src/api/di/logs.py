from dishka import Provider, Scope, provide
from logging import Logger

from logs import logger as _root_logger  # FIXME: глобалы ...


# FIXME: не зарегистрировал контейнер в __init__
class LoggerProvider(Provider):  # FIXME: надо его прокинуть в infra_provider
    scope = Scope.REQUEST

    app_logger: Logger = provide(source=lambda: _root_logger.getChild("app")    )
