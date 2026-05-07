"""initial idempotent

Revision ID: 001
Revises:
Create Date: 2026-05-07

"""
from alembic import op

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Create the enum type (only if not already present)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE investigationstatus AS ENUM (
                'OPEN', 'PENDING_APPROVAL', 'APPROVED', 'REJECTED', 'COMPLETED', 'FAILED'
            );
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # 2. Create the table using raw SQL (idempotent)
    op.execute("""
        CREATE TABLE IF NOT EXISTS investigations (
            id              TEXT PRIMARY KEY,
            created_at      TIMESTAMP,
            updated_at      TIMESTAMP,
            status          investigationstatus,
            drift_event     JSONB,
            triage_result   TEXT,
            proposed_action TEXT,
            final_decision  TEXT,
            checkpoint_id   TEXT,
            job_id          TEXT,
            model_version   TEXT
        );
    """)

def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS investigations")
    op.execute("DROP TYPE IF EXISTS investigationstatus")