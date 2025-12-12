from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017/" 
    DB_NAME: str = "safeiot_db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()