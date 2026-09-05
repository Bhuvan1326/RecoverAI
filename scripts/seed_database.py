#!/usr/bin/env python3
"""
One-command seed script (Section 36). Idempotent-ish: safe to re-run against
an empty DB; will error clearly on a claim_number collision if re-run
against an already-seeded DB (by design -- silently duplicating financial
records is worse than a clear error).

Usage:
    python scripts/seed_database.py
    python scripts/seed_database.py --claims 5000
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.models.domain import User, UserRole

REPO_ROOT = Path(__file__).resolve().parent.parent

DEMO_USERS = [
    ("admin@recoverai.demo", "Ava Admin", "DemoPass123!", UserRole.ADMIN),
    ("reviewer@recoverai.demo", "Rita Reviewer", "DemoPass123!", UserRole.REVIEWER),
    ("biller@recoverai.demo", "Ben Biller", "DemoPass123!", UserRole.BILLER),
    ("analyst@recoverai.demo", "Ana Analyst", "DemoPass123!", UserRole.ANALYST),
]


def seed_users():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        created = []
        for email, name, password, role in DEMO_USERS:
            existing = db.query(User).filter(User.email == email).first()
            if existing:
                continue
            db.add(User(email=email, full_name=name, hashed_password=hash_password(password), role=role))
            created.append(email)
        db.commit()
        print(f"Users: created {len(created)} new demo users (skipped {len(DEMO_USERS) - len(created)} already present).")
    finally:
        db.close()


def run(cmd: list[str], label: str):
    print(f"\n=== {label} ===")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"!! {label} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=int, default=3000)
    parser.add_argument("--payers", type=int, default=12)
    parser.add_argument("--providers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-training", action="store_true", help="Skip ML model training (faster iteration)")
    args = parser.parse_args()

    python = sys.executable

    seed_users()
    run(
        [python, "scripts/generate_data.py", "--claims", str(args.claims), "--payers", str(args.payers), "--providers", str(args.providers), "--seed", str(args.seed)],
        "Synthetic claims/denials/appeals",
    )
    run([python, "scripts/ingest_documents.py"], "RAG document ingestion")

    if not args.skip_training:
        run([python, "-m", "ml.training.train_denial_model"], "Train denial-risk models")

    print("\n=== Seed complete ===")
    print("Demo credentials (development only -- do not reuse anywhere real):")
    for email, name, password, role in DEMO_USERS:
        print(f"  {role.value:10s} {email:28s} / {password}")


if __name__ == "__main__":
    main()
