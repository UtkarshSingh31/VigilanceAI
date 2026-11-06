# app/dependencies.py
from fastapi import Request, HTTPException, status
from backend_ml.core.pipeline import PipelineOrchestrator

def get_pipeline_orchestrator(request: Request) -> PipelineOrchestrator:
    """
    Retrieves the global PipelineOrchestrator instance mounted in app.state during lifespan startup.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Inference Engine is not initialized or still booting."
        )
    return orchestrator

def get_alert_repository(request: Request):
    """
    Retrieves the Supabase alert repository from app.state.
    """
    repo = getattr(request.app.state, "alert_repo", None)
    return repo
