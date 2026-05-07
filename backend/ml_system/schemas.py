# backend/ml_system/schemas.py - Keep original clean names
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float      # clean name for API
    cons_price_idx: float    # clean name for API
    cons_conf_idx: float     # clean name for API
    euribor3m: float
    nr_employed: float       # clean name for API