"""FastAPI entrypoint for the agent system."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.agent_system.state.checkpoints import create_checkpointer, setup_checkpointer
from backend.agent_system.graph import create_state_graph
from backend.agent_system.webhook import router as webhook_router
from backend.agent_system.state.models import Base
from backend.agent_system.state.session import engine
from dotenv import load_dotenv
load_dotenv()  # load .env file from project root

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    saver = create_checkpointer()             # NO await
    await setup_checkpointer(saver)
    app.state.agent_graph = create_state_graph().compile(checkpointer=saver)
    yield

app = FastAPI(title="Agent System", version="0.2.0", lifespan=lifespan)

app.include_router(webhook_router)

@app.get("/health")
async def health():
    return {"status": "ok"}