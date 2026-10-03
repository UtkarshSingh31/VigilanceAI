from fastapi import APIRouter, status, Depends, BackgroundTasks, HTTPException
from typing import Dict, Any

from backend_ml.logger import get_logger
from backend_ml.core.pipeline import PipelineOrchestrator
from app.dependencies import get_pipeline_orchestrator, get_alert_repository
from app.schemas.telemetry import TelemetryBatchRequest, AnomalyResponse
from app.schemas.alert import IncidentRecord

logger = get_logger(__name__)

router = APIRouter(prefix="/telemetry", tags=["Telemetry Ingestion"])


async def persist_anomaly_task(
    alert_repo: Any,
    student_id: str,
    room_id: str,
    inference_result: Dict[str, Any],
    flagged_title: str
):
    """
    Background worker task to persist anomaly logs asynchronously to Supabase.
    Keeps the HTTP request-response cycle non-blocking and sub-second.
    """
    try:
        incident = IncidentRecord(
            reconstruction_loss=inference_result["reconstruction_loss"],
            effective_threshold=inference_result["effective_threshold"],
            point_anomaly_detected=inference_result["point_anomaly_detected"],
            sequence_anomaly_detected=(
                inference_result["reconstruction_loss"] > inference_result["effective_threshold"]
            ),
            flagged_window_title=flagged_title
        )
        
        # Async write to Supabase alert repository
        await alert_repo.save_incident(student_id=student_id, room_id=room_id, incident=incident)
        logger.info(f"Anomaly incident persisted asynchronously for student '{student_id}' in room '{room_id}'.")
    except Exception as e:
        logger.error(f"Failed to persist anomaly incident asynchronously: {e}")


@router.post(
    "",
    response_model=AnomalyResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest 30-second telemetry sequence and evaluate anomaly status"
)
async def ingest_telemetry_batch(
    payload: TelemetryBatchRequest,
    background_tasks: BackgroundTasks,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator),
    alert_repo: Any = Depends(get_alert_repository)
) -> AnomalyResponse:
    """
    Ingests exactly 30 records of raw telemetry, processes them through the ML pipeline,
    and returns immediate inference results.
    
    Database Cost Optimization Rule:
    - Normal sequences (is_anomaly == False) are discarded from RAM; zero DB writes.
    - Anomalous sequences (is_anomaly == True) trigger an async DB write to Supabase.
    """
    try:
        # Convert Pydantic records into the raw list of dictionaries expected by pipeline.py
        telemetry_dicts = [record.model_dump() for record in payload.records]

        # Execute in-memory ML inference pass
        result = orchestrator.process_telemetry_batch(
            room_id=payload.room_id,
            telemetry_data=telemetry_dicts,
            profile_multiplier=payload.profile_multiplier
        )

        is_anomaly = result.get("is_anomaly", False)

        # Decision Gate: Selective Asynchronous Persistence
        if is_anomaly:
            # Extract the last observed window title for situational explainability
            flagged_title = payload.records[-1].window_title

            background_tasks.add_task(
                persist_anomaly_task,
                alert_repo=alert_repo,
                student_id=payload.student_id,
                room_id=payload.room_id,
                inference_result=result,
                flagged_title=flagged_title
            )
            logger.warning(
                f"Anomaly detected for student '{payload.student_id}' in room '{payload.room_id}' | "
                f"Loss: {result['reconstruction_loss']:.4f} > Cutoff: {result['effective_threshold']:.4f} "
                f"(Point Anomaly: {result['point_anomaly_detected']})"
            )
        else:
            logger.debug(
                f"Batch normal for student '{payload.student_id}' in room '{payload.room_id}' | "
                f"Loss: {result['reconstruction_loss']:.4f} <= Cutoff: {result['effective_threshold']:.4f}. "
                f"Payload discarded from RAM."
            )

        return AnomalyResponse(
            room_id=payload.room_id,
            is_anomaly=is_anomaly,
            reconstruction_loss=result["reconstruction_loss"],
            effective_threshold=result["effective_threshold"],
            point_anomaly_detected=result["point_anomaly_detected"]
        )

    except KeyError as ke:
        logger.error(f"Configuration or key error during telemetry ingestion: {ke}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room configuration missing or invalid: {str(ke)}"
        )
    except ValueError as ve:
        logger.error(f"Validation error during telemetry inference: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid telemetry sequence contract: {str(ve)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error processing telemetry batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(e)}"
        )

    