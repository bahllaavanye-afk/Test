"""User model definitions.

This module defines the :class:`User` ORM model used throughout the
QuantEdge trading platform. The model includes basic authentication
fields, activity flags, and a relationship to the ``Account`` model.
"""

from __future__ import annotations

import uuid
from typing import List

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


class User(Base, TimestampMixin):
    """ORM representation of a system user.

    Attributes
    ----------
    id: str
        Primary key generated as a UUID string.
    email: str
        Unique e‑mail address used for login and communication.
    hashed_password: str
        Bcrypt‑hashed password; never stored in plain text.
    is_active: bool
        Indicates whether the user account is currently active.
    is_superuser: bool
        Flag for administrative privileges.
    accounts: List[Account]
        Collection of :class:`Account` objects owned by the user.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Primary key as a UUID string",
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User e‑mail address (unique)",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Bcrypt‑hashed user password",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Indicates if the user account is active",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Administrative privilege flag",
    )

    accounts: Mapped[List["Account"]] = relationship(
        "Account",
        back_populates="user",
        lazy="select",
        doc="Related accounts owned by the user",
    )