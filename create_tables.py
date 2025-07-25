from app.db.database import engine, Base
import app.db.models  # Tüm modellerin import edildiğinden emin olun

Base.metadata.create_all(bind=engine) 