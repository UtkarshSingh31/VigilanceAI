from typing import List, Dict, Any
from fastapi import APIRouter, status, Depends, HTTPException
from backend_ml.logger import get_logger
from app.schemas.room import RoomConfigCreate, RoomConfigResponse
from app.dependencies import get_pipeline_orchestrator
from backend_ml.core.pipeline import PipelineOrchestrator

logger = get_logger(__name__)

router = APIRouter(prefix="/rooms", tags=["Room Management"])


@router.post(
    "",
    response_model=RoomConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register or update room-specific ML artifacts"
)
async def register_room(
    config: RoomConfigCreate,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> RoomConfigResponse:
    """
    Registers a new classroom context or updates an existing room's
    HDBSCAN artifact path, cluster speed baseline map, and fallback speed.
    """
    try:
        # Dynamically mount room configuration into the running orchestrator in RAM
        orchestrator.room_configs[config.room_id] = {
            "clusterer_path": config.clusterer_path,
            "cluster_speed_map": config.cluster_speed_map,
            "global_train_mean": config.global_train_mean,
            "base_threshold": config.base_threshold
        }

        logger.info(f"Room context successfully registered/updated for ID: '{config.room_id}'")

        return RoomConfigResponse(
            room_id=config.room_id,
            room_name=config.room_name,
            clusterer_path=config.clusterer_path,
            global_train_mean=config.global_train_mean,
            base_threshold=config.base_threshold,
            is_active=True
        )

    except Exception as e:
        logger.exception(f"Failed to register room '{config.room_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not initialize room configuration: {str(e)}"
        )


@router.get(
    "",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="List all active configured room IDs"
)
async def list_active_rooms(
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> List[str]:
    """
    Returns a list of all room_ids currently loaded and ready
    for real-time telemetry inference in memory.
    """
    return list(orchestrator.room_configs.keys())


@router.get(
    "/{room_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch operational config for a specific room"
)
async def get_room_details(
    room_id: str,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> Dict[str, Any]:
    """
    Retrieves the in-memory artifact paths and base thresholds for a given room.
    """
    if room_id not in orchestrator.room_configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room '{room_id}' is not loaded or configured in the ML engine."
        )

    config = orchestrator.room_configs[room_id]
    return {
        "room_id": room_id,
        "clusterer_path": config["clusterer_path"],
        "global_train_mean": config["global_train_mean"],
        "base_threshold": config.get("base_threshold", 1.6923),
        "total_mapped_clusters": len(config.get("cluster_speed_map", {}))
    }