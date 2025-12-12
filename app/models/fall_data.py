from pydantic import Field
from app.models.base import MongoBaseModel

class FallData(MongoBaseModel):
    ax: float = Field(..., description="Aceleração no eixo X")
    ay: float = Field(..., description="Aceleração no eixo Y")
    az: float = Field(..., description="Aceleração no eixo Z")
    gx: float = Field(..., description="Giroscópio no eixo X")
    gy: float = Field(..., description="Giroscópio no eixo Y")
    gz: float = Field(..., description="Giroscópio no eixo Z")

class FallDataCreate(FallData):
    pass