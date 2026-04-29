from __future__ import annotations

import datetime
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

class Base(DeclarativeBase):
    pass

class TrafficReading(Base):

    __tablename__ = "traffic_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    intersection_id = Column(String(32), nullable=False, index=True)
    camera_id = Column(String(32), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)
    vehicle_count = Column(Integer, default=0)
    density = Column(Float, default=0.0)
    avg_speed = Column(Float, default=0.0)
    queue_length = Column(Float, default=0.0)
    flow_rate = Column(Float, default=0.0)
    congestion_level = Column(Float, default=0.0)

class SignalState(Base):

    __tablename__ = "signal_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    intersection_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)
    phase = Column(String(16), nullable=False)
    green_duration = Column(Float, nullable=False)
    cycle_length = Column(Float, nullable=False)
    optimized = Column(Integer, default=0)

class TrafficPrediction(Base):

    __tablename__ = "traffic_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    intersection_id = Column(String(32), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    horizon_minutes = Column(Integer, nullable=False)
    predicted_density = Column(Float)
    predicted_flow = Column(Float)
    predicted_queue = Column(Float)
    confidence = Column(Float)

class VideoJob(Base):

    __tablename__ = "video_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False)
    input_path = Column(Text, nullable=False)
    intersection_id = Column(String(32))
    status = Column(String(16), default="pending")
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)

class DatabaseManager:

    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = "sqlite:///data/traffic.db"
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self.engine = create_engine(db_url, connect_args=connect_args, echo=False)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

_db: Optional[DatabaseManager] = None
_db_url: Optional[str] = None

def get_db(db_url: Optional[str] = None) -> DatabaseManager:
    global _db, _db_url
    if _db is None or (db_url is not None and db_url != _db_url):
        _db_url = db_url
        _db = DatabaseManager(db_url)
    return _db
