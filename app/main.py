import uvicorn

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from starlette.middleware.cors import CORSMiddleware
from dishka.integrations.fastapi import setup_dishka

from config import Config
from src.api import container, router
from src.api.logs import RequestLoggingMiddleware


def create_fastapi_app() -> FastAPI:
    app = FastAPI(
        title="Сервис моделирования и прогнозирования",
        root_path=Config.APP_NGINX_PREFIX,
    )

    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.ALLOW_ORIGINS.split(),
        allow_credentials=bool(Config.ALLOW_CREDENTIALS),
        allow_methods=Config.ALLOW_METHODS.split(),
        allow_headers=Config.ALLOW_HEADERS.split(),
    )

    app.add_middleware(RequestLoggingMiddleware)

    app.openapi_schema = get_openapi(
        title="iep-forecast-service",
        version="1.0",
        routes=app.routes,
        servers=[{'url': Config.APP_NGINX_PREFIX}]
    )

    return app


app = create_fastapi_app()
setup_dishka(container, app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
