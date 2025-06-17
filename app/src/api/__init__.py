from fastapi import APIRouter

from .di import container
from .v1 import v1_router

router = APIRouter(prefix="api/")
router.include_router(v1_router)
