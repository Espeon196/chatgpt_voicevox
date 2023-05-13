from pydantic import BaseSettings, AnyHttpUrl
from dotenv import load_dotenv


class Settings(BaseSettings):
    voicevox_address: AnyHttpUrl


load_dotenv()
settings = Settings()