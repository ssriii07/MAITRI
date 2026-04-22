from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from .database import Base
from datetime import datetime

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, index=True) # "user" or "agent"
    content = Column(String)
    stress_score = Column(Float, nullable=True) # Overall inferred stress
    stress_tier = Column(String, nullable=True) # Low, Moderate, High
    shap_data = Column(Text, nullable=True) # JSON string of explainability features
    timestamp = Column(DateTime, default=datetime.utcnow)

class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    encrypted_content = Column(Text) # AES-256 encrypted
    mood_tag = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserBaseline(Base):
    __tablename__ = "user_baselines"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
