from backend.app.db.database import Base, engine
from backend.app.db import models  # noqa: F401


def init_db() -> None:
    """Create all application database tables."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")