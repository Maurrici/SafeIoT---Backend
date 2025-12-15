from pydantic import EmailStr, Field
from typing import List, Optional
from app.models.base import MongoBaseModel, PyObjectId

class Patient(MongoBaseModel):
    name: str = Field(..., min_length=3)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=6)
    phoneNumber: Optional[str] = None
    observer_ids: List[PyObjectId] = Field(default_factory=list)

class PatientCreate(Patient):
    class Config:
        exclude = {'id'}

class PatientLogin(MongoBaseModel):
    email: EmailStr
    password: str 