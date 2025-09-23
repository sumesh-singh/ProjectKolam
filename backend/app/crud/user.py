"""
CRUD operations for User model
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.crud.base import CRUDBase
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """
    CRUD operations for User model
    """

    async def get_by_email(self, db: AsyncSession, *, email: str) -> Optional[User]:
        """
        Get user by email
        """
        result = await db.execute(select(User).where(User.email == email))
        return result.scalars().first()

    async def get_by_username(self, db: AsyncSession, *, username: str) -> Optional[User]:
        """
        Get user by username
        """
        result = await db.execute(select(User).where(User.username == username))
        return result.scalars().first()

    async def update_last_login(self, db: AsyncSession, *, user_id: str) -> None:
        """
        Update user's last login timestamp
        """
        user = await self.get(db, id=user_id)
        if user:
            user.last_login = datetime.utcnow()
            await db.commit()

    async def is_active(self, user: User) -> bool:
        """
        Check if user is active
        """
        return user.is_active

    async def is_admin(self, user: User) -> bool:
        """
        Check if user is admin
        """
        return user.role == "admin"

    async def is_expert(self, user: User) -> bool:
        """
        Check if user is expert or admin
        """
        return user.role in ["expert", "admin"]


user_crud = CRUDUser(User)
