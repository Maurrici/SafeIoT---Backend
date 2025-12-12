from datetime import datetime
from typing import List
from pydantic import Field
from app.models.base import MongoBaseModel, PyObjectId
from app.models.fall_data import FallData

class FallRegister(MongoBaseModel):
    patient_id: PyObjectId = Field(..., description="ID do paciente que sofreu a queda")
    date: datetime = Field(default_factory=datetime.utcnow, description="Data e hora do registro")
    is_fall: bool = Field(default=False, description="True se o serviço de avaliação confirmou a queda")
    data_points: List[FallData] = Field(default_factory=list) 

class FallRegisterCreate(MongoBaseModel):
    patient_id: str = Field(..., description="ID do paciente (string)")
    is_fall: bool = Field(default=False)
    data_points: List[FallData]