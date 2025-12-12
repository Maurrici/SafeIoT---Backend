from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.DB_NAME]

patient_collection = db.get_collection("patients")
observer_collection = db.get_collection("observers")
fall_register_collection = db.get_collection("fall_registers")