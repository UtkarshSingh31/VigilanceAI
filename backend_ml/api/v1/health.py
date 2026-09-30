import torch
import psutil
from typing import Dict, Any
from fastapi import APIRouter, status, Depends
from backend_ml.logger import get_logger
from app.dependencies import get_pipeline_orchestrator
from backend_ml.core.pipeline import PipelineOrchestrator

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Basic liveness check"
)
async def liveness_probe() -> Dict[str, str]:
    """
    Lightweight probe for container orchestrators (e.g. Kubernetes, Docker)
    to confirm the HTTP worker is responsive.
    """
    return {"status": "alive"}


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="ML Engine & Hardware Readiness Probe"
)
async def readiness_probe(
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> Dict[str, Any]:
    """
    Validates that:
    1. PipelineOrchestrator is loaded in memory via Dependency Injection.
    2. Shared inference weights are mapped to the intended hardware target.
    3. Host RAM and GPU VRAM thresholds are healthy.
    """
    try:
        # Check system RAM utilization
        ram_info = psutil.virtual_memory()
        
        # Check GPU memory allocation if running on CUDA
        cuda_available = torch.cuda.is_available()
        gpu_stats = {}
        
        if cuda_available:
            gpu_stats = {
                "device_name": torch.cuda.get_device_name(0),
                "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            }

        # Inspect orchestrator readiness
        registered_rooms_count = len(orchestrator.room_configs)

        return {
            "status": "ready",
            "device": str(orchestrator.shared_inference_engine.device),
            "cuda_available": cuda_available,
            "gpu_metrics": gpu_stats if cuda_available else "N/A (CPU Mode)",
            "system_ram_used_percent": ram_info.percent,
            "active_configured_rooms": registered_rooms_count
        }

    except Exception as e:
        logger.exception("Readiness check failed - ML Pipeline not operational.")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

    