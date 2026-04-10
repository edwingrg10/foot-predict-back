from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Base de datos ──────────────────────────────────────────────────────────
    # En producción (Render/Railway) define DATABASE_URL directamente en variables de entorno.
    # En local sigue usando SQL Server si no se define DATABASE_URL.
    DATABASE_URL: Optional[str] = None

    # SQL Server local (solo se usa si DATABASE_URL no está definida)
    DB_SERVER: str = "."
    DB_NAME:   str = "FootballPredictor"
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"

    @property
    def db_url(self) -> str:
        if self.DATABASE_URL:
            # PostgreSQL en producción — Render/Supabase usan postgres://, SQLAlchemy necesita postgresql://
            url = self.DATABASE_URL
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            return url
        # SQL Server local (Windows Auth)
        driver = self.DB_DRIVER.replace(" ", "+")
        return (
            f"mssql+pyodbc://@{self.DB_SERVER}/{self.DB_NAME}"
            f"?driver={driver}&TrustServerCertificate=yes&Encrypt=no"
        )

    # ── Auth ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-this-to-a-very-secret-key-in-production"
    ALGORITHM:  str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h

    DEBUG: bool = False

    # OpenWeather (opcional)
    WEATHER_API_KEY: Optional[str] = None

    CORS_ORIGINS: list[str] = ["http://localhost:4200", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()
