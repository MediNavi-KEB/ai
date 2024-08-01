from sqlalchemy import Column, Integer, String, Enum
from sqlalchemy.dialects.mysql import ENUM
from sqlalchemy.orm import relationship
from db.connection import Base  # Base를 db.connection에서 가져옵니다.

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    name = Column(String(100), nullable=False)
    phone = Column(String(20))
    address = Column(String(255))
    gender = Column(ENUM('남성', '여성'))

    favorites = relationship("Favorite", back_populates="user")
