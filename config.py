from pydantic_settings import BaseSettings
from enums import ModelsEnum
class Settings(BaseSettings):
    API_TOKEN: str = "dummy_key"
    MODEL: ModelsEnum = ModelsEnum.LARGE
    NOTES_DIR: str = "my_notes"
    class Config:
        env_file = ".env"
settings = Settings()
