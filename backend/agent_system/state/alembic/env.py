import sys
import os
from logging.config import fileConfig

# Add the project root to sys.path so that 'backend' and 'shared' are importable.
# This file is at:  backend/agent_system/state/alembic/env.py
# The project root is four levels up (backend/agent_system/state/alembic → root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sqlalchemy import create_engine
from alembic import context

# Now you can import your models normally
from backend.agent_system.state.models import Base
target_metadata = Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise ValueError("Database URL not found in alembic config.")
    connectable = create_engine(url)
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()