import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    agent = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "agent": self.agent,
            "status": self.status,
            "result": json.loads(self.result_json) if self.result_json else None,
            "error": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def get_session() -> Session:
    return Session(engine)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
