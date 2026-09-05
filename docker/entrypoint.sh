#!/bin/sh
set -e

echo "Waiting for database..."
python - <<'PYEOF'
import time
import sys
sys.path.insert(0, "/repo/backend")
from sqlalchemy import create_engine, text
from app.core.config import get_settings

settings = get_settings()
for attempt in range(30):
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database is ready.")
        break
    except Exception as e:
        print(f"  ...database not ready yet ({e}); retrying ({attempt + 1}/30)")
        time.sleep(2)
else:
    print("Database never became ready.", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "Running Alembic migrations (upgrade head)..."
cd /repo/backend
python -m alembic upgrade head
cd /repo

exec "$@"
