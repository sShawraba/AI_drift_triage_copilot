"""
Self-contained LangGraph Postgres checkpointer using proper typed serialization.
Handles Interrupt / special objects by encoding them via JsonPlusSerializer.
"""
import json
import base64
from typing import Any, Optional, AsyncIterator, Sequence
import asyncpg
import uuid
import os
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@postgres:5432/mlops")

def _serialize(serde, obj: Any) -> str:
    """Serialize an object to a JSONB-safe string using the given serde."""
    encoding, data_bytes = serde.dumps_typed(obj)
    wrapper = {
        "encoding": encoding,
        "data": base64.b64encode(data_bytes).decode("ascii"),
    }
    return json.dumps(wrapper)

def _deserialize(serde, json_str: str) -> Any:
    """Reverse _serialize: parse the wrapper and use serde.loads_typed."""
    wrapper = json.loads(json_str)
    encoding = wrapper["encoding"]
    data_bytes = base64.b64decode(wrapper["data"])
    return serde.loads_typed((encoding, data_bytes))

class PostgresSaver(BaseCheckpointSaver):

    def __init__(self, conn_string: str):
        super().__init__(serde=JsonPlusSerializer())
        self.conn_string = conn_string
        self.pool: Optional[asyncpg.Pool] = None

    async def setup(self) -> None:
        self.pool = await asyncpg.create_pool(self.conn_string)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    checkpoint_data JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT now(),
                    PRIMARY KEY (thread_id, checkpoint_id)
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value JSONB,
                    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
                );
            """)

    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT checkpoint_id, checkpoint_data, metadata FROM agent_checkpoints "
                "WHERE thread_id = $1 ORDER BY created_at DESC LIMIT 1",
                thread_id,
            )
            if not row:
                return None
            checkpoint = _deserialize(self.serde, row["checkpoint_data"])
            metadata = _deserialize(self.serde, row["metadata"])
            writes = await conn.fetch(
                "SELECT task_id, idx, channel, value FROM checkpoint_writes "
                "WHERE thread_id = $1 AND checkpoint_id = $2 "
                "ORDER BY task_id, idx",
                thread_id, row["checkpoint_id"],
            )
            pending_writes = [
                (w["channel"], _deserialize(self.serde, w["value"]) if w["value"] else None)
                for w in writes
            ]
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": row["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                pending_writes=pending_writes,
            )

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = str(uuid.uuid4())
        checkpoint_json = _serialize(self.serde, checkpoint)
        metadata_json = _serialize(self.serde, metadata)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO agent_checkpoints (thread_id, checkpoint_id, checkpoint_data, metadata) "
                "VALUES ($1, $2, $3::jsonb, $4::jsonb)",
                thread_id,
                checkpoint_id,
                checkpoint_json,
                metadata_json,
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            return
        async with self.pool.acquire() as conn:
            for idx, (channel, value) in enumerate(writes):
                serialized_value = _serialize(self.serde, value) if value is not None else None
                await conn.execute(
                    "INSERT INTO checkpoint_writes (thread_id, checkpoint_id, task_id, idx, channel, value) "
                    "VALUES ($1, $2, $3, $4, $5, $6::jsonb) "
                    "ON CONFLICT (thread_id, checkpoint_id, task_id, idx) DO NOTHING",
                    thread_id,
                    checkpoint_id,
                    task_id,
                    idx,
                    channel,
                    serialized_value,
                )

    async def alist(
        self,
        config: Optional[dict],
        *,
        limit: Optional[int] = None,
        before: Optional[dict] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if not config:
            return
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return
        query = "SELECT checkpoint_id, checkpoint_data, metadata FROM agent_checkpoints WHERE thread_id = $1 ORDER BY created_at DESC"
        params = [thread_id]
        if limit is not None:
            query += f" LIMIT {limit}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            for row in rows:
                checkpoint = _deserialize(self.serde, row["checkpoint_data"])
                metadata = _deserialize(self.serde, row["metadata"])
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                )

    async def adelete_thread(self, thread_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM agent_checkpoints WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", thread_id)

    async def aget_next_version(self, current: Optional[str], channel: Any) -> str:
        return str(uuid.uuid4())


def create_checkpointer() -> PostgresSaver:
    return PostgresSaver(DATABASE_URL)

async def setup_checkpointer(saver: PostgresSaver):
    await saver.setup()