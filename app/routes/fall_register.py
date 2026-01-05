from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import List
from bson import ObjectId
from datetime import datetime
import os
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
from app.models.fall_register import FallRegister, FallRegisterCreate, FallRegisterResponse
from app.models.fall_data import FallData
from app.models.patient import Patient
from app.database import fall_register_collection, patient_collection, observer_collection

router = APIRouter(
    prefix="/fall-registers",
    tags=["Fall Registers"]
)

conf = ConnectionConfig(
    MAIL_USERNAME="safeiot2026@gmail.com",
    MAIL_PASSWORD="lqps uupd gusg eeqq",
    MAIL_FROM="safeiot2026@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

@router.post("/", response_model=FallRegister, status_code=status.HTTP_201_CREATED)
async def create_fall_register(
    register_data: FallRegisterCreate, 
    background_tasks: BackgroundTasks
):
    if not ObjectId.is_valid(register_data.patient_id):
        raise HTTPException(status_code=400, detail="Invalid Patient ID format.")
    
    pid = ObjectId(register_data.patient_id)
    patient = await patient_collection.find_one({"_id": pid})
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")

    fall_register_dict = register_data.model_dump(by_alias=True)
    fall_register_dict["patient_id"] = pid
    fall_register_dict["date"] = datetime.utcnow()
    
    if "_id" in fall_register_dict:
        del fall_register_dict["_id"]

    result = await fall_register_collection.insert_one(fall_register_dict)    
    new_register_doc = await fall_register_collection.find_one({"_id": result.inserted_id})
    
    # --- LÓGICA DE NOTIFICAÇÃO DOS OBSERVADORES ---
    if register_data.is_fall:
        observer_ids = patient.get("observer_ids", [])

        if observer_ids:
            # Busca os e-mails de todos os observadores cujos IDs estão na lista
            observers_cursor = observer_collection.find(
                {"_id": {"$in": observer_ids}},
                {"email": 1, "_id": 0}
            )
            
            emails_to_notify = [obs["email"] async for obs in observers_cursor if "email" in obs]

            if emails_to_notify:
                patient_name = patient.get("name", "Paciente")
                
                html_content = f"""
                <div style="font-family: sans-serif; border: 1px solid #dcdcdc; padding: 20px;">
                    <h1 style="color: #d32f2f;">🚨 ALERTA DE QUEDA DETECTADA</h1>
                    <p>Este é um aviso automático do sistema <b>SafeIoT</b>.</p>
                    <p>Uma queda foi detectada para o paciente sob sua observação: <b>{patient_name}</b>.</p>
                    <hr>
                    <p><b>Data do evento:</b> {fall_register_dict["date"].strftime('%d/%m/%Y às %H:%M:%S')}</p>
                    <p>Por favor, entre em contato ou desloque-se ao local imediatamente.</p>
                </div>
                """

                message = MessageSchema(
                    subject=f"URGENTE: Alerta de Queda - {patient_name}",
                    recipients=emails_to_notify,
                    body=html_content,
                    subtype=MessageType.html
                )

                fm = FastMail(conf)
                background_tasks.add_task(fm.send_message, message)

    return FallRegister(**new_register_doc)

@router.get("/", response_model=List[FallRegisterResponse]) 
async def get_patient_fall_history(patient_id: str):
    
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Formato de ID inválido."
        )
    
    pid = ObjectId(patient_id)

    patient_doc = await patient_collection.find_one({"_id": pid})
    
    if patient_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente não encontrado."
        )

    patient_obj = Patient(**patient_doc) 
    cursor = fall_register_collection.find({"patient_id": pid}).sort("date", -1)
    
    registers_response: List[FallRegisterResponse] = []
    async for doc in cursor:
        doc['patient'] = patient_obj 
        registers_response.append(FallRegisterResponse(**doc))
        
    return registers_response