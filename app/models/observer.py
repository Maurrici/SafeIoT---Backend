from pydantic import EmailStr, Field, BaseModel
from typing import List, Optional
from app.models.base import MongoBaseModel, PyObjectId

class Observer(MongoBaseModel):
    name: str = Field(..., min_length=3)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=6)
    phoneNumber: Optional[str] = None

    patient_ids: List[PyObjectId] = Field(default_factory=list)

    telegram_chat_id: Optional[int] = Field(
        default=None,
        description="Chat ID do Telegram para envio de alertas em tempo real"
    )

class ObserverCreate(Observer):
    class Config:
        exclude = {'id'}

class ObserverLogin(BaseModel):
    email: EmailStr
    password: str
