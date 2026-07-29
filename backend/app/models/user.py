import uuid
import logging
from typing import List, Optional

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin

logger = logging.getLogger(__name__)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    accounts: Mapped[List["Account"]] = relationship(
        "Account", back_populates="user", lazy="select"
    )

    def __init__(
        self,
        email: str,
        hashed_password: str,
        is_active: bool = True,
        is_superuser: bool = False,
        accounts: Optional[List["Account"]] = None,
    ) -> None:
        """
        Initialise a User instance with validation for edge‑case inputs.

        Args:
            email: User e‑mail address. Must be a non‑empty string.
            hashed_password: Hashed password string. Must be non‑empty.
            is_active: Flag indicating if the user is active.
            is_superuser: Flag indicating if the user has super‑user privileges.
            accounts: Optional list of Account objects. If ``None`` an empty list is used.

        Raises:
            ValueError: If ``email`` or ``hashed_password`` are ``None`` or empty.
        """
        if not email:
            raise ValueError("email must be provided and non‑empty")
        if not hashed_password:
            raise ValueError("hashed_password must be provided and non‑empty")

        self.email = email
        self.hashed_password = hashed_password
        self.is_active = is_active
        self.is_superuser = is_superuser
        # Ensure accounts is always a list; handle None gracefully
        self.accounts = accounts if accounts is not None else []

    def add_account(self, account: "Account") -> None:
        """
        Safely add an Account to the user.

        Args:
            account: An Account instance to associate with the user.

        Raises:
            ValueError: If ``account`` is ``None``.
        """
        if account is None:
            raise ValueError("account cannot be None")
        self.accounts.append(account)
        logger.debug("Added account %s to user %s", getattr(account, "id", None), self.id)

    def get_account_by_index(self, index: int) -> "Account":
        """
        Retrieve an account by its positional index, handling off‑by‑one errors.

        Args:
            index: Zero‑based index of the desired account.

        Returns:
            The Account instance at the given index.

        Raises:
            TypeError: If ``index`` is not an integer.
            IndexError: If ``index`` is out of bounds.
        """
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= len(self.accounts):
            raise IndexError("account index out of range")
        return self.accounts[index]

    def list_accounts(self) -> List["Account"]:
        """
        Return a shallow copy of the accounts list to prevent external mutation.

        Returns:
            A list of Account objects associated with the user.
        """
        return list(self.accounts)