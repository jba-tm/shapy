from typing import List, Optional, Union
from pathlib import Path
from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator


class Settings(BaseSettings):
    # Dirs
    BASE_DIR: Optional[str] = Path(__file__).resolve().parent.parent.parent.as_posix()
    PROJECT_DIR: Optional[str] = Path(__file__).resolve().parent.parent.as_posix()
    # Project
    VERSION: Optional[str] = '0.1.0'
    DEBUG: Optional[bool] = False
    PAGINATION_MAX_SIZE: Optional[int] = 25

    DOMAIN: Optional[str] = 'localhost:8000'
    ENABLE_SSL: Optional[bool] = False
    SITE_URL: Optional[str] = 'http://localhost'
    ROOT_PATH: Optional[str] = ""
    ROOT_PATH_IN_SERVERS: Optional[bool] = True
    OPENAPI_URL: Optional[str] = '/openapi.json'

    API_V1_STR: Optional[str] = "/api/v1"
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    SERVER_NAME: Optional[str] = "Project name"
    SERVER_HOST: Optional[AnyHttpUrl] = "http://localhost"
    BACKEND_CORS_ORIGINS: Optional[List[AnyHttpUrl]] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    PROJECT_NAME: Optional[str] = 'project_name'

    FIRST_SUPERUSER: Optional[EmailStr] = 'admin@example.com'
    FIRST_SUPERUSER_PASSWORD: Optional[str] = 'change_this'

    MEAS_DEFINITION_PATH: Optional[str] = "data/utility_files/measurements/measurement_defitions.yaml"
    MEAS_VERTICES_PATH: Optional[str] = "data/utility_files/measurements/smplx_measurements.yaml"
    SMPL_MODEL_PATH: Optional[str] = "data/body_models"

    class Config:
        case_sensitive = True
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
