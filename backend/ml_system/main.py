from fastapi import FastAPI
from backend.ml_system.routers import predict

app = FastAPI()

app.include_router(predict.router)