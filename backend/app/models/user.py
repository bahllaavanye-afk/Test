import uuid
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


class User(Base, TimestampMixin):
    """SQLAlchemy ORM model representing an application user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for the user (UUID4).",
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="User's email address, used for login and notifications.",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Cryptographically hashed password (e.g., bcrypt).",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Indicates whether the user account is active.",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Flag for administrative privileges.",
    )

    accounts: Mapped[List["Account"]] = relationship(
        "Account", back_populates="user", lazy="select"
    )


# ---------------------------------------------------------------------------
# Pydantic schemas for API interaction
# ---------------------------------------------------------------------------


class UserBase(BaseModel):
    """Base schema shared by multiple user representations."""

    email: EmailStr = Field(
        ...,
        description="User's email address.",
        example="alice@example.com",
    )
    is_active: Optional[bool] = Field(
        default=True,
        description="Whether the user account is active.",
        example=True,
    )
    is_superuser: Optional[bool] = Field(
        default=False,
        description="Administrative privilege flag.",
        example=False,
    )

    @validator("email")
    def email_must_be_lowercase(cls, v: str) -> str:
        """Enforce lowercase email for consistency."""
        return v.lower()


class UserCreate(UserBase):
    """Schema used when creating a new user."""

    password: str = Field(
        ...,
        min_length=8,
        description="Plain-text password; will be hashed before storage.",
        example="Str0ngP@ssw0rd!",
    )

    @validator("password")
    def password_strength(cls, v: str) -> str:
        """Basic password strength validation."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserRead(UserBase):
    """Schema returned when reading user information."""

    id: str = Field(
        ...,
        description="Unique identifier for the user.",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    created_at: Optional[str] = Field(
        None,
        description="Timestamp when the user was created (ISO 8601).",
        example="2023-01-01T12:00:00Z",
    )
    updated_at: Optional[str] = Field(
        None,
        description="Timestamp of the last update to the user record (ISO 8601).",
        example="2023-01-02T08:30:00Z",
    )

    class Config:
        orm_mode = True


class UserUpdate(BaseModel):
    """Schema used for partial updates to a user."""

    email: Optional[EmailStr] = Field(
        None,
        description="New email address for the user.",
        example="bob@example.com",
    )
    password: Optional[str] = Field(
        None,
        min_length=8,
        description="New password; will be hashed before storage.",
        example="N3wP@ssw0rd!",
    )
    is_active: Optional[bool] = Field(
        None,
        description="Set to false to deactivate the account.",
        example=False,
    )
    is_superuser: Optional[bool] = Field(
        None,
        description="Grant or revoke administrative privileges.",
        example=True,
    )

    @validator("password")
    def password_strength(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


__all__ = [
    "User",
    "UserBase",
    "UserCreate",
    "UserRead",
    "UserUpdate",
]