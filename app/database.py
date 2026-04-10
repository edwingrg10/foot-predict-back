from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from .config import settings
from .models import Base


_is_postgres = settings.db_url.startswith("postgresql")

engine = create_engine(
    settings.db_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    # PostgreSQL en Render/Supabase free tier tiene límite de conexiones simultáneas
    pool_size=5 if _is_postgres else 10,
    max_overflow=10 if _is_postgres else 20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("[DB] Tablas verificadas/creadas en FootballPredictor")
