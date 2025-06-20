import uvicorn

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from dishka.integrations.fastapi import setup_dishka

from config import Config
from src.api import container, router
from src.api.logs import RequestLoggingMiddleware


def create_fastapi_app() -> FastAPI:
    app = FastAPI(title="Сервис моделирования и прогнозирования")
    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.ALLOW_ORIGINS.split(),
        allow_credentials=bool(Config.ALLOW_CREDENTIALS),
        allow_methods=Config.ALLOW_METHODS.split(),
        allow_headers=Config.ALLOW_HEADERS.split(),
    )

    app.add_middleware(RequestLoggingMiddleware)

    return app


def create_app():
    app = create_fastapi_app()
    setup_dishka(container, app)
    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
