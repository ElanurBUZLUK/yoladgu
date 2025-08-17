from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import joinedload

from app.repositories.base_repository import BaseRepository
from app.models.user import User, UserRole, LearningStyle
from app.schemas.user import UserCreate, UserUpdate


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """User repository - Kullanıcı işlemleri için özel metodlar"""
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Email ile kullanıcı getir"""
        return await self.get_by_field(db, "email", email)
    
    async def get_by_username(self, db: AsyncSession, username: str) -> Optional[User]:
        """Username ile kullanıcı getir"""
        return await self.get_by_field(db, "username", username)
    
    async def get_active_users(
        self, 
        db: AsyncSession, 
        role: Optional[UserRole] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Aktif kullanıcıları getir"""
        filters = {"is_active": "true"}
        if role:
            filters["role"] = role
        
        return await self.get_multi(db, skip=skip, limit=limit, filters=filters)
    
    async def get_students_by_level(
        self,
        db: AsyncSession,
        subject: str,
        level: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Belirli seviyedeki öğrencileri getir"""
        filters = {
            "role": UserRole.STUDENT,
            "is_active": "true"
        }
        
        if subject == "math":
            filters["current_math_level"] = level
        elif subject == "english":
            filters["current_english_level"] = level
        
        return await self.get_multi(db, skip=skip, limit=limit, filters=filters)
    
    async def get_students_by_learning_style(
        self,
        db: AsyncSession,
        learning_style: LearningStyle,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Öğrenme stiline göre öğrencileri getir"""
        filters = {
            "role": UserRole.STUDENT,
            "is_active": "true",
            "learning_style": learning_style
        }
        
        return await self.get_multi(db, skip=skip, limit=limit, filters=filters)
    
    async def search_users(
        self,
        db: AsyncSession,
        search_term: str,
        role: Optional[UserRole] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Kullanıcı arama"""
        search_fields = ["username", "email"]
        
        # Arama sorgusu
        query = select(self.model).where(
            and_(
                self.model.is_active == "true",
                or_(
                    self.model.username.ilike(f"%{search_term}%"),
                    self.model.email.ilike(f"%{search_term}%")
                )
            )
        )
        
        # Rol filtresi
        if role:
            query = query.where(self.model.role == role)
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_user_with_attempts(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 10
    ) -> Optional[User]:
        """Kullanıcıyı denemeleri ile birlikte getir"""
        from app.models.student_attempt import StudentAttempt
        
        result = await db.execute(
            select(User)
            .options(joinedload(User.student_attempts))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_statistics(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Kullanıcı istatistiklerini getir"""
        from sqlalchemy import func
        from app.models.student_attempt import StudentAttempt
        
        # Toplam deneme sayısı
        result = await db.execute(
            select(func.count(StudentAttempt.id))
            .where(StudentAttempt.user_id == user_id)
        )
        total_attempts = result.scalar() or 0
        
        # Doğru cevap sayısı
        result = await db.execute(
            select(func.count(StudentAttempt.id))
            .where(
                and_(
                    StudentAttempt.user_id == user_id,
                    StudentAttempt.is_correct == True
                )
            )
        )
        correct_attempts = result.scalar() or 0
        
        # Doğruluk oranı
        accuracy_rate = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0
        
        # Ortalama zorluk seviyesi
        result = await db.execute(
            select(func.avg(StudentAttempt.difficulty_level))
            .where(StudentAttempt.user_id == user_id)
        )
        avg_difficulty = result.scalar() or 0
        
        # Ortalama süre
        result = await db.execute(
            select(func.avg(StudentAttempt.time_spent))
            .where(StudentAttempt.user_id == user_id)
        )
        avg_time = result.scalar() or 0
        
        return {
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy_rate": round(accuracy_rate, 2),
            "avg_difficulty": round(avg_difficulty, 2),
            "avg_time_spent": round(avg_time, 2)
        }
    
    async def update_user_level(
        self,
        db: AsyncSession,
        user_id: str,
        subject: str,
        new_level: int
    ) -> Optional[User]:
        """Kullanıcı seviyesini güncelle"""
        user = await self.get(db, user_id)
        if not user:
            return None
        
        if subject == "math":
            user.current_math_level = new_level
        elif subject == "english":
            user.current_english_level = new_level
        
        await db.commit()
        await db.refresh(user)
        return user
    
    async def deactivate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Kullanıcıyı deaktive et"""
        user = await self.get(db, user_id)
        if not user:
            return False
        
        user.is_active = "false"
        await db.commit()
        return True
    
    async def activate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Kullanıcıyı aktive et"""
        user = await self.get(db, user_id)
        if not user:
            return False
        
        user.is_active = "true"
        await db.commit()
        return True
    
    async def get_users_by_role(
        self,
        db: AsyncSession,
        role: UserRole,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Rol bazında kullanıcıları getir"""
        filters = {"role": role, "is_active": "true"}
        return await self.get_multi(db, skip=skip, limit=limit, filters=filters)
    
    async def get_user_count_by_role(self, db: AsyncSession) -> Dict[str, int]:
        """Rol bazında kullanıcı sayılarını getir"""
        from sqlalchemy import func
        
        result = await db.execute(
            select(
                User.role,
                func.count(User.id).label('count')
            )
            .where(User.is_active == "true")
            .group_by(User.role)
        )
        
        counts = {}
        for row in result.all():
            counts[row.role.value] = row.count
        
        return counts
    
    async def get_top_performers(
        self,
        db: AsyncSession,
        subject: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """En iyi performans gösteren öğrencileri getir"""
        from sqlalchemy import func
        from app.models.student_attempt import StudentAttempt
        
        # Performans hesaplama
        result = await db.execute(
            select(
                StudentAttempt.user_id,
                func.count(StudentAttempt.id).label('total_attempts'),
                func.sum(func.case((StudentAttempt.is_correct == True, 1), else_=0)).label('correct_attempts'),
                func.avg(StudentAttempt.difficulty_level).label('avg_difficulty')
            )
            .where(
                and_(
                    StudentAttempt.subject == subject,
                    StudentAttempt.user_id.in_(
                        select(User.id).where(
                            and_(
                                User.role == UserRole.STUDENT,
                                User.is_active == "true"
                            )
                        )
                    )
                )
            )
            .group_by(StudentAttempt.user_id)
            .having(func.count(StudentAttempt.id) >= 5)  # En az 5 deneme
            .order_by(
                func.sum(func.case((StudentAttempt.is_correct == True, 1), else_=0)).desc()
            )
            .limit(limit)
        )
        
        performers = []
        for row in result.all():
            accuracy = (row.correct_attempts / row.total_attempts * 100) if row.total_attempts > 0 else 0
            
            # Kullanıcı bilgilerini al
            user = await self.get(db, row.user_id)
            if user:
                performers.append({
                    "user_id": str(row.user_id),
                    "username": user.username,
                    "email": user.email,
                    "total_attempts": row.total_attempts,
                    "correct_attempts": row.correct_attempts,
                    "accuracy_rate": round(accuracy, 2),
                    "avg_difficulty": round(row.avg_difficulty, 2)
                })
        
        return performers


# Global user repository instance
user_repository = UserRepository()
