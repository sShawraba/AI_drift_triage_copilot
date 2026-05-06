"""Synchronous database operations for investigations."""
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.agent_system.state.models import Investigation, InvestigationStatus

def create_investigation(db: Session, investigation_id: str, drift_event: dict) -> Investigation:
    inv = Investigation(
        id=investigation_id,
        drift_event=drift_event,
        status=InvestigationStatus.OPEN,
        model_version=drift_event.get("model_version"),
    )
    db.add(inv)
    db.commit()
    db.refresh(inv)
    return inv

def get_investigation(db: Session, investigation_id: str) -> Optional[Investigation]:
    return db.query(Investigation).filter(Investigation.id == investigation_id).first()

def update_status(db: Session, investigation_id: str, status: InvestigationStatus):
    inv = get_investigation(db, investigation_id)
    if inv:
        inv.status = status
        db.commit()

def update_checkpoint_id(db: Session, investigation_id: str, checkpoint_id: str):
    inv = get_investigation(db, investigation_id)
    if inv:
        inv.checkpoint_id = checkpoint_id
        db.commit()

def update_investigation_results(
    db: Session,
    investigation_id: str,
    triage_result: Optional[str],
    proposed_action: Optional[str],
    final_decision: Optional[str] = None,
    status: InvestigationStatus = InvestigationStatus.COMPLETED,
):
    inv = get_investigation(db, investigation_id)
    if inv:
        if triage_result:
            inv.triage_result = triage_result
        if proposed_action is not None:
            inv.proposed_action = proposed_action
        if final_decision:
            inv.final_decision = final_decision
        inv.status = status
        db.commit()

def list_open_investigations(db: Session) -> List[Investigation]:
    return db.query(Investigation).filter(
        Investigation.status.in_([InvestigationStatus.OPEN, InvestigationStatus.PENDING_APPROVAL])
    ).all()

def list_pending_approvals(db: Session) -> List[Investigation]:
    return db.query(Investigation).filter(Investigation.status == InvestigationStatus.PENDING_APPROVAL).all()