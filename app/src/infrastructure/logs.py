from logging import getLogger

logger = getLogger("root-logger")
logger.info("Логирование инициализировано", extra={"correlation_id": "startup"})
