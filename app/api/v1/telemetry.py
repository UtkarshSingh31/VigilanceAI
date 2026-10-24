from typing import Any
from fastapi import APIRouter, status, Depends, BackgroundTasks, HTTPException

from backend_ml.logger import get_logger
from backend_ml.core.pipeline import PipelineOrchestrator
from app.dependencies import get_pipeline_orchestrator, get_alert_repository
from app.schemas.telemetry import TelemetryBatchRequest
from app.schemas.alert import IncidentRecord

logger = get_logger(__name__)

router = APIRouter(prefix="/telemetry", tags=["Telemetry Ingestion"])


async def persist_anomaly_task(
    alert_repo: Any,
    student_id: str,
    room_id: str,
    incident: IncidentRecord
):
    try:
        if alert_repo:
            await alert_repo.save_incident(student_id=student_id, room_id=room_id, incident=incident)
            logger.info(f"Anomaly incident persisted for student '{student_id}' in room '{room_id}'.")
    except Exception as e:
        logger.error(f"Failed to persist anomaly incident asynchronously: {e}")


@router.post(
    "",
    response_model=IncidentRecord,
    status_code=status.HTTP_200_OK,
    summary="Ingest 30-second telemetry sequence"
)
async def ingest_telemetry_batch(
    payload: TelemetryBatchRequest,
    background_tasks: BackgroundTasks,
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator),
    alert_repo: Any = Depends(get_alert_repository)
) -> IncidentRecord:
    try:
        telemetry_dicts = [record.model_dump() for record in payload.records]

        result = orchestrator.process_telemetry_batch(
            room_id=payload.room_id,
            telemetry_data=telemetry_dicts,
            profile_multiplier=payload.profile_multiplier
        )

        is_point = result.get("point_anomaly_detected", False)
        loss = result.get("reconstruction_loss", 0.0)
        cutoff = result.get("effective_threshold", 1.4066)
        is_sequence = loss > cutoff

        incident = IncidentRecord(
            reconstruction_loss=loss,
            effective_threshold=cutoff,
            point_anomaly_detected=is_point,
            sequence_anomaly_detected=is_sequence,
            flagged_window_title=payload.records[-1].window_title if (is_point or is_sequence) else None
        )

        # Zero-Cost Rule: Only persist when a threshold breach occurs
        if is_point or is_sequence:
            background_tasks.add_task(
                persist_anomaly_task,
                alert_repo=alert_repo,
                student_id=payload.student_id,
                room_id=payload.room_id,
                incident=incident
            )

        return incident

    except KeyError as ke:
        logger.error(f"Configuration key error: {ke}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ke))
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))