# models.py
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, JSON, func
from sqlalchemy.orm import relationship
from db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)          # doc_id (UUID string)
    filename = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    drafts = relationship("Draft", back_populates="document", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="document", cascade="all, delete-orphan")
    finals = relationship("FinalExtract", back_populates="document", cascade="all, delete-orphan")

class Draft(Base):
    __tablename__ = "drafts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    payload = Column(JSON, nullable=False)         # full JSON draft
    source = Column(String, nullable=True)         # e.g. "grobid+llm"
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="drafts")

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    reviewer = Column(String, nullable=False)      # "A", "B", or user id/email
    payload = Column(JSON, nullable=False)         # reviewer’s JSON selections/edits
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="reviews")

class FinalExtract(Base):
    __tablename__ = "final_extracts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    payload = Column(JSON, nullable=False)         # adjudicated final JSON
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="finals")
