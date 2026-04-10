from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # SQL Server — autenticación Windows (Trusted_Connection)
    DB_SERVER: str = "."
    DB_NAME: str = "FootballPredictor"
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"

    @property
    def DATABASE_URL(self) -> str:
        driver = self.DB_DRIVER.replace(" ", "+")
        return (
            f"mssql+pyodbc://@{self.DB_SERVER}/{self.DB_NAME}"
            f"?driver={driver}&TrustServerCertificate=yes&Encrypt=no"
        )

    SECRET_KEY: str = "change-this-to-a-very-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h
    DEBUG: bool = False

    # OpenWeather (opcional)
    WEATHER_API_KEY: Optional[str] = None

    CORS_ORIGINS: list[str] = ["http://localhost:4200", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()
