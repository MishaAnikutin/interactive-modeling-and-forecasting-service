from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync


model_predict_router = APIRouter(prefix="/model_predicting", tags=["Прогнозы моделей"])
