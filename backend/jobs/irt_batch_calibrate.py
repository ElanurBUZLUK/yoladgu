"""
IRT Batch Calibration Job
Performs batch calibration of IRT models from recent attempt data
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db
from ml.math.irt import irt_model, ItemParams, StudentAbility

logger = structlog.get_logger()

class IRTBatchCalibrator:
    """Batch calibrator for IRT models"""
    
    def __init__(self):
        self.model_path = "backend/data/ml/irt_model.json"
        self.min_attempts_per_item = 10
        self.min_attempts_per_student = 5
        self.calibration_iterations = 5
        
    async def load_attempt_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Load attempt data from database
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of attempt records
        """
        try:
            async with get_db() as db:
                # Query attempts table (assuming it exists)
                query_sql = text("""
                    SELECT 
                        user_id,
                        item_id,
                        correct,
                        duration,
                        skills,
                        predicted_p,
                        created_at
                    FROM attempts 
                    WHERE created_at >= NOW() - INTERVAL :days DAY
                    AND item_id IS NOT NULL
                    AND correct IS NOT NULL
                    ORDER BY created_at DESC
                """)
                
                result = await db.execute(query_sql, {"days": days})
                rows = result.fetchall()
                
                attempts = []
                for row in rows:
                    attempts.append({
                        "user_id": row.user_id,
                        "item_id": row.item_id,
                        "correct": row.correct,
                        "duration": row.duration,
                        "skills": row.skills,
                        "predicted_p": row.predicted_p,
                        "created_at": row.created_at
                    })
                
                logger.info("Loaded attempt data", count=len(attempts), days=days)
                return attempts
                
        except Exception as e:
            logger.error("Failed to load attempt data", error=str(e))
            return []
    
    async def prepare_training_data(self, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare training data from attempts
        
        Args:
            attempts: List of attempt records
            
        Returns:
            Prepared training data
        """
        logger.info("Preparing training data", attempts_count=len(attempts))
        
        # Group attempts by item and student
        item_attempts = {}
        student_attempts = {}
        
        for attempt in attempts:
            item_id = attempt["item_id"]
            user_id = attempt["user_id"]
            
            if item_id not in item_attempts:
                item_attempts[item_id] = []
            if user_id not in student_attempts:
                student_attempts[user_id] = []
            
            item_attempts[item_id].append(attempt)
            student_attempts[user_id].append(attempt)
        
        # Filter items and students with sufficient data
        valid_items = {item_id: attempts for item_id, attempts in item_attempts.items() 
                      if len(attempts) >= self.min_attempts_per_item}
        valid_students = {user_id: attempts for user_id, attempts in student_attempts.items() 
                         if len(attempts) >= self.min_attempts_per_student}
        
        logger.info("Filtered training data", 
                   valid_items=len(valid_items), valid_students=len(valid_students))
        
        return {
            "item_attempts": valid_items,
            "student_attempts": valid_students,
            "total_attempts": len(attempts)
        }
    
    async def initialize_irt_model(self, training_data: Dict[str, Any]) -> bool:
        """
        Initialize IRT model with training data
        
        Args:
            training_data: Prepared training data
            
        Returns:
            Success status
        """
        try:
            logger.info("Initializing IRT model")
            
            # Add students to IRT model
            for user_id in training_data["student_attempts"].keys():
                if user_id not in irt_model.students:
                    student = StudentAbility(user_id=user_id, theta=0.0)
                    irt_model.add_student(student)
            
            # Add items to IRT model (with default parameters)
            for item_id in training_data["item_attempts"].keys():
                if item_id not in irt_model.items:
                    # Extract skills from attempts if available
                    skills = set()
                    for attempt in training_data["item_attempts"][item_id]:
                        if attempt.get("skills"):
                            if isinstance(attempt["skills"], list):
                                skills.update(attempt["skills"])
                            elif isinstance(attempt["skills"], str):
                                skills.add(attempt["skills"])
                    
                    skill_weights = {skill: 1.0/len(skills) for skill in skills} if skills else None
                    
                    item_params = ItemParams(
                        item_id=item_id,
                        a=1.0,  # Default discrimination
                        b=0.0,  # Default difficulty
                        c=0.0,  # Default guessing
                        skill_weights=skill_weights
                    )
                    irt_model.add_item(item_params)
            
            # Add responses to IRT model
            for attempt in training_data["item_attempts"]:
                for item_id, attempts in attempt.items():
                    for attempt_data in attempts:
                        irt_model.add_response(
                            user_id=attempt_data["user_id"],
                            item_id=attempt_data["item_id"],
                            correct=attempt_data["correct"],
                            response_time=attempt_data.get("duration"),
                            skills=attempt_data.get("skills")
                        )
            
            logger.info("IRT model initialized", 
                       students=len(irt_model.students),
                       items=len(irt_model.items),
                       responses=len(irt_model.responses))
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize IRT model", error=str(e))
            return False
    
    async def calibrate_model(self) -> Dict[str, Any]:
        """
        Calibrate the IRT model
        
        Returns:
            Calibration metrics
        """
        try:
            logger.info("Starting IRT model calibration")
            
            # Perform calibration
            irt_model.calibrate_model(
                max_iterations=self.calibration_iterations,
                tolerance=0.01
            )
            
            # Calculate calibration metrics
            metrics = await self._calculate_calibration_metrics()
            
            logger.info("IRT model calibration completed", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error("Failed to calibrate IRT model", error=str(e))
            return {}
    
    async def _calculate_calibration_metrics(self) -> Dict[str, Any]:
        """Calculate calibration metrics"""
        try:
            # Calculate log loss
            log_loss = 0.0
            brier_score = 0.0
            total_predictions = 0
            
            for response in irt_model.responses:
                user_id = response["user_id"]
                item_id = response["item_id"]
                actual = 1.0 if response["correct"] else 0.0
                
                if user_id in irt_model.students and item_id in irt_model.items:
                    student = irt_model.students[user_id]
                    item = irt_model.items[item_id]
                    
                    # Calculate predicted probability
                    if item.skill_weights and student.skill_thetas:
                        predicted = irt_model.p_correct_multi_skill(student.skill_thetas, item)
                    else:
                        predicted = irt_model.p_correct(student.theta, item)
                    
                    # Avoid log(0)
                    predicted = max(min(predicted, 0.999), 0.001)
                    
                    # Log loss
                    log_loss -= actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
                    
                    # Brier score
                    brier_score += (predicted - actual) ** 2
                    
                    total_predictions += 1
            
            if total_predictions > 0:
                log_loss /= total_predictions
                brier_score /= total_predictions
            
            # Item statistics
            item_stats = {}
            for item_id, item in irt_model.items.items():
                item_stats[item_id] = {
                    "discrimination": item.a,
                    "difficulty": item.b,
                    "guessing": item.c,
                    "skill_weights": item.skill_weights
                }
            
            # Student statistics
            student_stats = {}
            for user_id, student in irt_model.students.items():
                student_stats[user_id] = {
                    "theta": student.theta,
                    "skill_thetas": student.skill_thetas
                }
            
            return {
                "log_loss": log_loss,
                "brier_score": brier_score,
                "total_predictions": total_predictions,
                "items_count": len(irt_model.items),
                "students_count": len(irt_model.students),
                "responses_count": len(irt_model.responses),
                "item_stats": item_stats,
                "student_stats": student_stats
            }
            
        except Exception as e:
            logger.error("Failed to calculate calibration metrics", error=str(e))
            return {}
    
    async def save_calibrated_model(self) -> bool:
        """Save calibrated model to file and database"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save to file
            irt_model.save_model(self.model_path)
            
            # Update database items (if items table exists)
            await self._update_database_items()
            
            logger.info("Calibrated model saved", path=self.model_path)
            return True
            
        except Exception as e:
            logger.error("Failed to save calibrated model", error=str(e))
            return False
    
    async def _update_database_items(self):
        """Update database items with calibrated IRT parameters"""
        try:
            async with get_db() as db:
                for item_id, item in irt_model.items.items():
                    # Update items table with IRT parameters
                    update_sql = text("""
                        UPDATE items_math 
                        SET 
                            irt_discrimination = :a,
                            irt_difficulty = :b,
                            irt_guessing = :c,
                            irt_skill_weights = :skill_weights,
                            irt_calibrated_at = NOW()
                        WHERE id = :item_id
                    """)
                    
                    await db.execute(update_sql, {
                        "a": item.a,
                        "b": item.b,
                        "c": item.c,
                        "skill_weights": json.dumps(item.skill_weights) if item.skill_weights else None,
                        "item_id": item_id
                    })
                
                await db.commit()
                logger.info("Database items updated with IRT parameters")
                
        except Exception as e:
            logger.warning("Failed to update database items", error=str(e))
    
    async def run_calibration(self, days: int = 30) -> Dict[str, Any]:
        """
        Run complete calibration pipeline
        
        Args:
            days: Number of days to look back for training data
            
        Returns:
            Calibration results
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting IRT batch calibration", days=days)
            
            # Load attempt data
            attempts = await self.load_attempt_data(days)
            if not attempts:
                return {"error": "No attempt data found", "days": days}
            
            # Prepare training data
            training_data = await self.prepare_training_data(attempts)
            if not training_data["item_attempts"] or not training_data["student_attempts"]:
                return {"error": "Insufficient training data", "attempts": len(attempts)}
            
            # Initialize IRT model
            if not await self.initialize_irt_model(training_data):
                return {"error": "Failed to initialize IRT model"}
            
            # Calibrate model
            calibration_metrics = await self.calibrate_model()
            if not calibration_metrics:
                return {"error": "Failed to calibrate model"}
            
            # Save calibrated model
            save_success = await self.save_calibrated_model()
            
            # Prepare results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                "success": True,
                "duration_seconds": duration,
                "training_data": {
                    "total_attempts": training_data["total_attempts"],
                    "valid_items": len(training_data["item_attempts"]),
                    "valid_students": len(training_data["student_attempts"])
                },
                "calibration_metrics": calibration_metrics,
                "model_saved": save_success,
                "model_path": self.model_path
            }
            
            logger.info("IRT batch calibration completed", results=results)
            return results
            
        except Exception as e:
            logger.error("IRT batch calibration failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }

# Global calibrator instance
irt_calibrator = IRTBatchCalibrator()

async def main():
    """Main calibration function"""
    try:
        results = await irt_calibrator.run_calibration(days=30)
        
        print("\n" + "="*60)
        print("IRT BATCH CALIBRATION RESULTS")
        print("="*60)
        
        if results.get("success"):
            print(f"✅ Calibration completed successfully")
            print(f"Duration: {results['duration_seconds']:.2f} seconds")
            print(f"Training data: {results['training_data']['total_attempts']} attempts")
            print(f"Valid items: {results['training_data']['valid_items']}")
            print(f"Valid students: {results['training_data']['valid_students']}")
            
            metrics = results['calibration_metrics']
            print(f"\nCalibration Metrics:")
            print(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
            print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
            print(f"  Items: {metrics.get('items_count', 0)}")
            print(f"  Students: {metrics.get('students_count', 0)}")
            print(f"  Responses: {metrics.get('responses_count', 0)}")
            
            print(f"\nModel saved: {results['model_saved']}")
            print(f"Model path: {results['model_path']}")
            
        else:
            print(f"❌ Calibration failed: {results.get('error', 'Unknown error')}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        logger.error("Calibration failed", error=str(e))

if __name__ == "__main__":
    asyncio.run(main())
