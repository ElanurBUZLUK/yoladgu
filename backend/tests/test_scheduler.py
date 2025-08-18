from unittest.mock import MagicMock
from app.workers.scheduler import Scheduler

def test_scheduler_tasks():
    mock_db = MagicMock()
    scheduler = Scheduler(mock_db)
    
    # Test lock acquisition
    mock_db.execute.return_value.scalar.return_value = True
    assert scheduler._acquire_lock() is True
    
    # Test task execution
    scheduler.update_spaced_repetition_due()
    scheduler.refresh_peer_hardness()
    scheduler.conditionally_swap_index()
    
    # Should have multiple execute calls
    assert mock_db.execute.call_count > 3