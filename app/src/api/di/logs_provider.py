from dishka import Provider, Scope, provide
from logging import Logger, getLogger


class LogsProvider(Provider):
    @provide(scope=Scope.APP)
    def logger(self) -> Logger:
        logger = getLogger("root-logger")
        logger.info("Логирование инициализировано", extra={"correlation_id": "startup"})
        return logger
