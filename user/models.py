from sqlalchemy import Column, Integer, String

from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True,index=True)
    name = Column(String(30), nullable=False)
    email = Column(String(40), nullable=False, unique=True)
    face_encoding = Column(String, nullable=False)
    