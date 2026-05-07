"""Drift event webhook and HIL endpoints."""
import uuid
from typing import Optional
import redis
from datetime import datetime
from fastapi import APIRouter, Depends, Request, HTTPException, Body
from sqlalchemy.orm import Session
from backend.agent_system.state.session import get_db
from backend.agent_system.state.repo import (
    create_investigation,
    update_investigation_results,
    get_investigation,
    list_pending_approvals,
    list_investigations,
)
from backend.agent_system.state.models import InvestigationStatus
from shared.schemas import DriftEvent
from backend.agent_system.graph import AgentState
from langgraph.types import Command

router = APIRouter(prefix="/webhook", tags=["webhook"])

@router.post("/drift-event")
async def receive_drift_event(event: DriftEvent, request: Request, db: Session = Depends(get_db)):
    investigation_id = str(uuid.uuid4())
    drift_dict = event.dict()
    if isinstance(drift_dict.get("timestamp"), datetime):
        drift_dict["timestamp"] = drift_dict["timestamp"].isoformat()
    create_investigation(db, investigation_id, drift_dict)

    initial_state: AgentState = {
        "investigation_id": investigation_id,
        "drift_event": drift_dict,
        "triage_result": "",
        "proposed_action": "",
        "final_decision": "",
        "next_step": "",
        "human_approval": False,
        "approval_decision": None,
    }
    graph = request.app.state.agent_graph
    config = {"configurable": {"thread_id": investigation_id}}
    try:
        result = await graph.ainvoke(initial_state, config)

        # Handle pause for human approval
        if result.get("__interrupt__"):
            # Extract proposed_action from the structured interrupt payload
            interrupts = result["__interrupt__"]
            proposed_action = ""
            if interrupts:
                val = interrupts[0].value
                if isinstance(val, dict):
                    proposed_action = val.get("proposed_action", "")
                else:
                    # Fallback parsing for plain string (backward compatibility)
                    if "Proposed action: " in val:
                        proposed_action = val.split("Proposed action: ")[1].split(".")[0].strip()

            update_investigation_results(
                db,
                investigation_id,
                triage_result=result.get("triage_result", ""),
                proposed_action=proposed_action,
                status=InvestigationStatus.PENDING_APPROVAL,
            )
            return {
                "investigation_id": investigation_id,
                "status": "pending_approval",
                "proposed_action": proposed_action,
            }

        # Normal completion (no interrupt)
        update_investigation_results(
            db,
            investigation_id,
            triage_result=result.get("triage_result", ""),
            proposed_action=result.get("proposed_action", ""),
            final_decision=result.get("final_decision", ""),
            status=InvestigationStatus.COMPLETED,
        )
        return {
            "investigation_id": investigation_id,
            "status": "completed",
            "final_state": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pending")
async def list_pending(db: Session = Depends(get_db)):
    invs = list_pending_approvals(db)
    return [
        {
            "id": inv.id,
            "model_version": inv.model_version,
            "proposed_action": inv.proposed_action,
            "triage_result": inv.triage_result,
            "created_at": inv.created_at.isoformat(),
        }
        for inv in invs
    ]

@router.post("/investigations/{investigation_id}/approve")
async def approve_investigation(
    investigation_id: str,
    request: Request,
    db: Session = Depends(get_db),
    approve: bool = Body(True, embed=True),
):
    inv = get_investigation(db, investigation_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")
    if inv.status != InvestigationStatus.PENDING_APPROVAL:
        raise HTTPException(status_code=400, detail="Investigation is not pending approval")

    graph = request.app.state.agent_graph
    config = {"configurable": {"thread_id": investigation_id}}
    command = Command(resume={"approved": approve})
    try:
        final_state = await graph.ainvoke(command, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    new_status = InvestigationStatus.COMPLETED if approve else InvestigationStatus.REJECTED
    update_investigation_results(
        db,
        investigation_id,
        triage_result=final_state.get("triage_result") or inv.triage_result or "",
        proposed_action=final_state.get("proposed_action") or inv.proposed_action or "",
        final_decision=final_state.get("final_decision") or inv.final_decision or "Action completed",
        status=new_status,
    )
    return {
        "investigation_id": investigation_id,
        "status": new_status.value,
        "final_state": final_state,
    }

@router.post("/investigations/{investigation_id}/reject")
async def reject_investigation(investigation_id: str, request: Request, db: Session = Depends(get_db)):
    return await approve_investigation(investigation_id, request, db, approve=False)

@router.get("/checkpoint/{investigation_id}")
async def get_checkpoint(investigation_id: str, request: Request):
    graph = request.app.state.agent_graph
    config = {"configurable": {"thread_id": investigation_id}}
    checkpoint_tuple = await graph.checkpointer.aget_tuple(config)
    if checkpoint_tuple:
        return {
            "checkpoint_id": checkpoint_tuple.config["configurable"]["checkpoint_id"],
            "state": checkpoint_tuple.checkpoint,
        }
    return {"error": "No checkpoint found"}



@router.get("/investigations/{investigation_id}")
async def get_single_investigation(
    investigation_id: str,
    db: Session = Depends(get_db),
):
    """Return one investigation by ID."""
    inv = get_investigation(db, investigation_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Not Found")
    return {
        "id": inv.id,
        "model_version": inv.model_version,
        "status": inv.status.value,
        "triage_result": inv.triage_result,
        "proposed_action": inv.proposed_action,
        "final_decision": inv.final_decision,
        "created_at": inv.created_at.isoformat(),
    }




@router.get("/investigations")
async def list_investigations_route(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Return investigations, filtered by status if provided."""
    invs = list_investigations(db, status)
    return [
        {
            "id": inv.id,
            "model_version": inv.model_version,
            "status": inv.status.value,
            "triage_result": inv.triage_result,
            "proposed_action": inv.proposed_action,
            "final_decision": inv.final_decision,
            "created_at": inv.created_at.isoformat(),
        }
        for inv in invs
    ]

@router.get("/queue/stats")
async def queue_stats():
    """Return job queue statistics (depth, DLQ, processed count)."""
    import redis, os
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return {
        "queue_depth": r.llen("agent:job_queue"),
        "dead_letter_depth": r.llen("agent:dead_letter_queue"),
        "processed_count": r.scard("agent:processed_keys"),
    }
    
    
    
