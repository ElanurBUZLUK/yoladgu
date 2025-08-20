import io, os, json, zipfile, datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from app.core.database import get_db
from app.core.security import verify_token

router = APIRouter(prefix="/api/v1/privacy", tags=["privacy"]) 

@router.post("/export")
def export_my_data(user_id: str = Depends(verify_token)):
    db = next(get_db())
    attempts = db.execute(text("SELECT * FROM attempts WHERE user_id=:u ORDER BY created_at DESC"), {"u": user_id}).mappings().all()
    vocab = db.execute(text("SELECT * FROM student_vocab_profile WHERE student_id=:u"), {"u": user_id}).mappings().all()
    events = db.execute(text("SELECT * FROM events WHERE payload->>'user_id' = :u ORDER BY ts DESC"), {"u": user_id}).mappings().all()

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('attempts.jsonl', "\n".join(json.dumps(dict(r)) for r in attempts))
        z.writestr('vocab_profile.jsonl', "\n".join(json.dumps(dict(r)) for r in vocab))
        z.writestr('events.jsonl', "\n".join(json.dumps(dict(r)) for r in events))
    mem.seek(0)
    db.execute(text("INSERT INTO privacy_requests(user_id, kind, status, result_path) VALUES(:u,'export','done',:p)"), {"u": user_id, "p": f"inmem:{datetime.datetime.utcnow().isoformat()}"})
    db.commit()

    from fastapi.responses import StreamingResponse
    filename = f"export_{user_id}_{int(datetime.datetime.utcnow().timestamp())}.zip"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)

@router.post("/delete")
def delete_my_data(user_id: str = Depends(verify_token)):
    db = next(get_db())
    db.execute(text("UPDATE attempts SET user_id = concat('anon-',id) WHERE user_id=:u"), {"u": user_id})
    db.execute(text("DELETE FROM student_vocab_profile WHERE student_id=:u"), {"u": user_id})
    db.execute(text("INSERT INTO privacy_requests(user_id, kind, status) VALUES(:u,'delete','done')"), {"u": user_id})
    db.commit()
    return {"status": "ok"}