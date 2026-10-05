from fastapi import APIRouter
from app.api.v1 import health, rooms, telemetry

# Master Version 1 API Router
api_v1_router = APIRouter(prefix="/api/v1")

# Mount sub-routers under their respective endpoint paths
api_v1_router.include_router(health.router)
api_v1_router.include_router(rooms.router)
api_v1_router.include_router(telemetry.router)

