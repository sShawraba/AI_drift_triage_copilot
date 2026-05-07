"""SQLAlchemy models for agent state."""
import datetime
import enum
from sqlalchemy import Column, String, DateTime, JSON, Enum as SAEnum
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class InvestigationStatus(str, enum.Enum):
    OPEN = "open"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"

class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(String, primary_key=True)                     # UUID
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    status = Column(SAEnum(InvestigationStatus), default=InvestigationStatus.OPEN)
    drift_event = Column(JSON)                                # raw DriftEvent dict
    triage_result = Column(String, nullable=True)
    proposed_action = Column(String, nullable=True)
    final_decision = Column(String, nullable=True)
    checkpoint_id = Column(String, nullable=True)             # LangGraph thread_id
    job_id = Column(String, nullable=True)                    # Redis queue job id
    model_version = Column(String, nullable=True)             # extracted from drift_event