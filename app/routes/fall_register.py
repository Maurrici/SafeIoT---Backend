# app/routes/fall_register.py

from fastapi import APIRouter, HTTPException, status
from typing import List
from bson import ObjectId
from datetime import datetime

from app.models.fall_register import FallRegister, FallRegisterCreate
from app.models.fall_data import FallData
from app.database import fall_register_collection, patient_collection

router = APIRouter(
    prefix="/fall-registers",
    tags=["Fall Registers"]
)

@router.post("/", response_model=FallRegister, status_code=status.HTTP_201_CREATED)
async def create_fall_register(register_data: FallRegisterCreate):
    """
    Cadastra um novo registro de evento, contendo dados brutos de sensores.
    """
    
    if not ObjectId.is_valid(register_data.patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid Patient ID format."
        )
        
    pid = ObjectId(register_data.patient_id)
    
    if await patient_collection.find_one({"_id": pid}) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Patient with ID {register_data.patient_id} not found."
        )

    fall_register_dict = register_data.model_dump(by_alias=True)
    
    fall_register_dict["patient_id"] = pid
    
    fall_register_dict["date"] = datetime.utcnow()
    
    if "_id" in fall_register_dict:
        del fall_register_dict["_id"]

    result = await fall_register_collection.insert_one(fall_register_dict)    
    new_register = await fall_register_collection.find_one({"_id": result.inserted_id})
    
    if new_register is None:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve inserted fall register."
        )
        
    return FallRegister(**new_register)


@router.get("/", response_model=List[FallRegister])
async def get_patient_fall_history(patient_id: str):
    
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Formato de ID inválido."
        )
    
    pid = ObjectId(patient_id)

    if await patient_collection.find_one({"_id": pid}) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente não encontrado."
        )

    cursor = fall_register_collection.find({"patient_id": pid}).sort("date", -1)
    
    registers = []

    async for doc in cursor:
        registers.append(FallRegister(**doc))
        
    return registers