from pydantic_settings import BaseSettings

from enums import ModelsEnum



class Settings(BaseSettings):
    API_TOKEN: str
    MODEL: ModelsEnum
    NOTES_DIR: str

    class Config:
        env_file = ".env"


settings = Settings()