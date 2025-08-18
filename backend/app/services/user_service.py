from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_, or_
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from app.models.user import User, UserRole, LearningStyle
from app.schemas.auth import UserCreate, UserUpdate, UserSearchQuery
from app.core.security import security_service
from app.core.cache import cache_service
import uuid
import secrets
from datetime import datetime, timedelta


class UserService:
    """User management service"""
    
    def __init__(self):
        pass
    
    async def create_user(self, db: AsyncSession, user_data: UserCreate) -> User:
        """Create a new user"""
        
        # Check if username or email already exists
        existing_user = await db.execute(
            select(User).where(
                (User.username == user_data.username) | 
                (User.email == user_data.email)
            )
        )
        
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        # Hash password
        hashed_password = security_service.get_password_hash(user_data.password)
        
        # Create user
        db_user = User(
            id=uuid.uuid4(),
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            role=user_data.role,
            learning_style=user_data.learning_style,
            current_math_level=1,
            current_english_level=1,
            is_active="true"
        )
        
        try:
            db.add(db_user)
            await db.commit()
            await db.refresh(db_user)
            return db_user
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
    
    async def authenticate_user(self, db: AsyncSession, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        
        result = await db.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not security_service.verify_password(password, user.hashed_password):
            return None
        
        if user.is_active != "true":
            return None
        
        return user
    
    async def get_user_by_id(self, db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def get_user_by_username(self, db: AsyncSession, username: str) -> Optional[User]:
        """Get user by username"""
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()
    
    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    async def update_user(self, db: AsyncSession, user_id: str, user_update: UserUpdate) -> Optional[User]:
        """Update user information"""
        
        # Get current user
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return None
        
        # Prepare update data
        update_data = {}
        
        if user_update.username is not None:
            # Check if new username is already taken
            existing_user = await db.execute(
                select(User).where(
                    (User.username == user_update.username) & 
                    (User.id != user_id)
                )
            )
            if existing_user.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
            update_data["username"] = user_update.username
        
        if user_update.email is not None:
            # Check if new email is already taken
            existing_user = await db.execute(
                select(User).where(
                    (User.email == user_update.email) & 
                    (User.id != user_id)
                )
            )
            if existing_user.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already taken"
                )
            update_data["email"] = user_update.email
        
        if user_update.learning_style is not None:
            update_data["learning_style"] = user_update.learning_style
        
        if user_update.current_math_level is not None:
            update_data["current_math_level"] = user_update.current_math_level
        
        if user_update.current_english_level is not None:
            update_data["current_english_level"] = user_update.current_english_level
        
        if not update_data:
            return user
        
        # Update user
        try:
            await db.execute(
                update(User).where(User.id == user_id).values(**update_data)
            )
            await db.commit()
            
            # Return updated user
            return await self.get_user_by_id(db, user_id)
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already taken"
            )
    
    async def update_user_levels(
        self, 
        db: AsyncSession, 
        user_id: str, 
        math_level: Optional[int] = None,
        english_level: Optional[int] = None
    ) -> Optional[User]:
        """Update user subject levels"""
        
        update_data = {}
        
        if math_level is not None:
            if not (1 <= math_level <= 5):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Math level must be between 1 and 5"
                )
            update_data["current_math_level"] = math_level
        
        if english_level is not None:
            if not (1 <= english_level <= 5):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="English level must be between 1 and 5"
                )
            update_data["current_english_level"] = english_level
        
        if not update_data:
            return await self.get_user_by_id(db, user_id)
        
        await db.execute(
            update(User).where(User.id == user_id).values(**update_data)
        )
        await db.commit()
        
        return await self.get_user_by_id(db, user_id)
    
    async def change_password(self, db: AsyncSession, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password"""
        
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return False
        
        # Verify current password
        if not security_service.verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_hashed_password = security_service.get_password_hash(new_password)
        
        # Update password
        await db.execute(
            update(User).where(User.id == user_id).values(hashed_password=new_hashed_password)
        )
        await db.commit()
        
        return True
    
    async def deactivate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Deactivate user account"""
        
        await db.execute(
            update(User).where(User.id == user_id).values(is_active="false")
        )
        await db.commit()
        
        return True
    
    async def activate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Activate user account"""
        
        await db.execute(
            update(User).where(User.id == user_id).values(is_active="true")
        )
        await db.commit()
        
        return True
    
    async def get_users(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100,
        role: Optional[UserRole] = None
    ) -> List[User]:
        """Get list of users with pagination"""
        
        query = select(User)
        
        if role:
            query = query.where(User.role == role)
        
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def search_users(
        self, 
        db: AsyncSession, 
        search_query: UserSearchQuery,
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """Search users with advanced filters"""
        
        query = select(User)
        conditions = []
        
        if search_query.query:
            # Search in username and email
            search_term = f"%{search_query.query}%"
            conditions.append(
                or_(
                    User.username.ilike(search_term),
                    User.email.ilike(search_term)
                )
            )
        
        if search_query.role:
            conditions.append(User.role == search_query.role)
        
        if search_query.learning_style:
            conditions.append(User.learning_style == search_query.learning_style)
        
        if search_query.min_level:
            conditions.append(
                or_(
                    User.current_math_level >= search_query.min_level,
                    User.current_english_level >= search_query.min_level
                )
            )
        
        if search_query.max_level:
            conditions.append(
                and_(
                    User.current_math_level <= search_query.max_level,
                    User.current_english_level <= search_query.max_level
                )
            )
        
        if search_query.is_active is not None:
            active_value = "true" if search_query.is_active else "false"
            conditions.append(User.is_active == active_value)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_user_count(self, db: AsyncSession) -> int:
        """Get total user count"""
        result = await db.execute(select(func.count(User.id)))
        return result.scalar()
    
    async def get_user_stats_summary(self, db: AsyncSession) -> Dict[str, Any]:
        """Get user statistics summary"""
        
        # Total users
        total_users = await self.get_user_count(db)
        
        # Active users
        active_result = await db.execute(
            select(func.count(User.id)).where(User.is_active == "true")
        )
        active_users = active_result.scalar()
        
        # Users by role
        role_stats = {}
        for role in UserRole:
            role_result = await db.execute(
                select(func.count(User.id)).where(User.role == role)
            )
            role_stats[role.value] = role_result.scalar()
        
        # Users by learning style
        learning_style_stats = {}
        for style in LearningStyle:
            style_result = await db.execute(
                select(func.count(User.id)).where(User.learning_style == style)
            )
            learning_style_stats[style.value] = style_result.scalar()
        
        # Recent registrations (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_result = await db.execute(
            select(func.count(User.id)).where(User.created_at >= week_ago)
        )
        recent_registrations = recent_result.scalar()
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "role_distribution": role_stats,
            "learning_style_distribution": learning_style_stats,
            "recent_registrations": recent_registrations
        }
    
    async def generate_password_reset_token(self, db: AsyncSession, email: str) -> Optional[str]:
        """Generate password reset token"""
        
        user = await self.get_user_by_email(db, email)
        if not user:
            return None
        
        # Generate secure token
        reset_token = secrets.token_urlsafe(32)
        
        # Store token in cache with 1 hour expiration
        cache_key = f"password_reset:{reset_token}"
        await cache_service.set(cache_key, str(user.id), expire=3600)
        
        return reset_token
    
    async def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify password reset token and return user ID"""
        
        cache_key = f"password_reset:{token}"
        user_id = await cache_service.get(cache_key)
        
        if user_id:
            # Delete token after use
            await cache_service.delete(cache_key)
            return user_id
        
        return None
    
    async def reset_password_with_token(self, db: AsyncSession, token: str, new_password: str) -> bool:
        """Reset password using token"""
        
        user_id = await self.verify_password_reset_token(token)
        if not user_id:
            return False
        
        # Hash new password
        new_hashed_password = security_service.get_password_hash(new_password)
        
        # Update password
        await db.execute(
            update(User).where(User.id == user_id).values(hashed_password=new_hashed_password)
        )
        await db.commit()
        
        return True
    
    async def bulk_user_operation(self, db: AsyncSession, user_ids: List[str], operation: str) -> Dict[str, Any]:
        """Perform bulk operations on users"""
        
        if operation not in ["activate", "deactivate", "delete"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid operation. Must be 'activate', 'deactivate', or 'delete'"
            )
        
        success_count = 0
        failed_ids = []
        
        for user_id in user_ids:
            try:
                if operation == "activate":
                    await self.activate_user(db, user_id)
                elif operation == "deactivate":
                    await self.deactivate_user(db, user_id)
                elif operation == "delete":
                    # Soft delete by deactivating
                    await self.deactivate_user(db, user_id)
                
                success_count += 1
            except Exception:
                failed_ids.append(user_id)
        
        return {
            "operation": operation,
            "total_requested": len(user_ids),
            "successful": success_count,
            "failed": len(failed_ids),
            "failed_ids": failed_ids
        }
    
    async def get_user_activity_summary(self, db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """Get user activity summary with real analytics data"""
        
        from app.models.student_attempt import StudentAttempt
        from sqlalchemy import func, and_, desc
        from datetime import datetime, timedelta
        
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return {}
        
        # Get total attempts and correct attempts
        attempts_result = await db.execute(
            select(
                func.count(StudentAttempt.id).label('total_attempts'),
                func.count(StudentAttempt.id).filter(StudentAttempt.is_correct == True).label('correct_attempts'),
                func.avg(StudentAttempt.response_time).label('avg_response_time')
            ).where(StudentAttempt.user_id == user_id)
        )
        attempts_stats = attempts_result.first()
        
        total_attempts = attempts_stats.total_attempts or 0
        correct_attempts = attempts_stats.correct_attempts or 0
        accuracy_rate = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0.0
        average_time_spent = attempts_stats.avg_response_time or 0.0
        
        # Get current streak
        streak_result = await db.execute(
            select(StudentAttempt.created_at)
            .where(StudentAttempt.user_id == user_id)
            .order_by(desc(StudentAttempt.created_at))
            .limit(1)
        )
        last_attempt = streak_result.scalar_one_or_none()
        
        current_streak = 0
        if last_attempt:
            # Calculate streak by checking consecutive days with attempts
            current_date = datetime.utcnow().date()
            last_attempt_date = last_attempt.date()
            
            if (current_date - last_attempt_date).days <= 1:
                # Check for consecutive days
                streak_date = last_attempt_date
                while True:
                    day_attempts = await db.execute(
                        select(StudentAttempt.id)
                        .where(
                            and_(
                                StudentAttempt.user_id == user_id,
                                func.date(StudentAttempt.created_at) == streak_date
                            )
                        )
                    )
                    if day_attempts.first():
                        current_streak += 1
                        streak_date -= timedelta(days=1)
                    else:
                        break
        
        # Get last activity
        last_activity = last_attempt.isoformat() if last_attempt else None
        
        # Get recent performance (last 10 attempts)
        recent_performance = await db.execute(
            select(StudentAttempt.is_correct)
            .where(StudentAttempt.user_id == user_id)
            .order_by(desc(StudentAttempt.created_at))
            .limit(10)
        )
        recent_correct = sum(1 for row in recent_performance.fetchall() if row[0])
        recent_accuracy = (recent_correct / 10 * 100) if recent_performance.rowcount > 0 else 0.0
        
        return {
            "user_id": str(user.id),
            "username": user.username,
            "role": user.role.value,
            "learning_style": user.learning_style.value,
            "math_level": user.current_math_level,
            "english_level": user.current_english_level,
            "account_created": user.created_at.isoformat(),
            "last_updated": user.updated_at.isoformat(),
            "is_active": user.is_active == "true",
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy_rate": round(accuracy_rate, 2),
            "average_time_spent": round(average_time_spent, 2),
            "current_streak": current_streak,
            "last_activity": last_activity,
            "recent_accuracy": round(recent_accuracy, 2),
            "recent_attempts": min(10, total_attempts)
        }


# Global user service instance
user_service = UserService()