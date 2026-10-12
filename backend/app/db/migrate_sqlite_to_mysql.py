"""Copy existing SQLite application rows into an empty configured MySQL DB."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.engine import Engine

from backend.app.db.database import Base, get_engine
from backend.app.db import models  # noqa: F401


TABLE_ORDER = (
    "users",
    "sessions",
    "searches",
    "search_results",
    "audit_log",
)
SQLITE_SOURCE_PATH = Path(__file__).resolve().parents[2] / "storage" / "app.db"


def _row_counts(engine: Engine) -> dict[str, int]:
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for table_name in TABLE_ORDER:
            table = Base.metadata.tables[table_name]
            counts[table_name] = int(
                connection.scalar(select(func.count()).select_from(table)) or 0
            )
    return counts


def _read_rows(engine: Engine, table_name: str) -> list[dict]:
    table = Base.metadata.tables[table_name]
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(table)).mappings()]


def _verify_rows(source: Engine, target: Engine) -> None:
    for table_name in TABLE_ORDER:
        source_rows = _read_rows(source, table_name)
        target_rows = _read_rows(target, table_name)
        if len(source_rows) != len(target_rows):
            raise RuntimeError(f"Migrated row count mismatch in {table_name}.")

        source_rows.sort(key=lambda row: row["id"])
        target_rows.sort(key=lambda row: row["id"])
        columns = tuple(Base.metadata.tables[table_name].columns.keys())
        for source_row, target_row in zip(source_rows, target_rows):
            for column in columns:
                source_value, target_value = source_row[column], target_row[column]
                if (
                    isinstance(source_value, float)
                    and isinstance(target_value, (float, int))
                ):
                    equal = math.isclose(
                        source_value,
                        float(target_value),
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    )
                else:
                    equal = source_value == target_value
                if not equal:
                    raise RuntimeError(
                        f"Migrated value mismatch in {table_name}.{column} "
                        f"for primary key {source_row['id']}."
                    )


def migrate(source_path: Path, target_engine: Engine) -> tuple[dict[str, int], dict[str, int]]:
    """Transfer rows with primary keys intact and verify relationships/counts."""
    if target_engine.dialect.name != "mysql":
        raise RuntimeError("The migration destination must be a MySQL database.")
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite source database does not exist: {source_path}")

    sqlite_engine = create_engine(f"sqlite:///{source_path.resolve().as_posix()}")
    try:
        source_inspector = inspect(sqlite_engine)
        missing = set(TABLE_ORDER) - set(source_inspector.get_table_names())
        if missing:
            raise RuntimeError(
                "SQLite source is missing expected tables: "
                + ", ".join(sorted(missing))
            )

        Base.metadata.create_all(bind=target_engine)
        before = _row_counts(sqlite_engine)
        target_before = _row_counts(target_engine)
        nonempty = {name: count for name, count in target_before.items() if count}
        if nonempty:
            raise RuntimeError(
                "MySQL destination must be empty to avoid overwriting or duplicating "
                f"data; existing row counts: {nonempty}"
            )

        with target_engine.begin() as target_connection, sqlite_engine.connect() as source_connection:
            for table_name in TABLE_ORDER:
                table = Base.metadata.tables[table_name]
                rows = source_connection.execute(select(table)).mappings().all()
                if rows:
                    target_connection.execute(table.insert(), [dict(row) for row in rows])

        after = _row_counts(target_engine)
        if before != after:
            raise RuntimeError(
                f"Migration row-count mismatch: SQLite={before}, MySQL={after}"
            )

        _verify_rows(sqlite_engine, target_engine)
        _verify_foreign_keys(target_engine)
        return before, after
    finally:
        sqlite_engine.dispose()


def _verify_foreign_keys(engine: Engine) -> None:
    checks = (
        ("sessions", "user_id", "users", "id"),
        ("searches", "user_id", "users", "id"),
        ("search_results", "search_id", "searches", "id"),
        ("audit_log", "user_id", "users", "id"),
    )
    with engine.connect() as connection:
        for child_name, child_key, parent_name, parent_key in checks:
            child = Base.metadata.tables[child_name]
            parent = Base.metadata.tables[parent_name]
            orphan_query = (
                select(func.count())
                .select_from(child.outerjoin(parent, child.c[child_key] == parent.c[parent_key]))
                .where(parent.c[parent_key].is_(None))
            )
            orphan_count = int(connection.scalar(orphan_query) or 0)
            if orphan_count:
                raise RuntimeError(
                    f"MySQL migration found {orphan_count} orphan rows in {child_name}."
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=SQLITE_SOURCE_PATH,
        help="Existing SQLite database file (default: backend/storage/app.db)",
    )
    args = parser.parse_args()

    before, after = migrate(args.source, get_engine())
    print("SQLite row counts:", before)
    print("MySQL row counts:  ", after)
    print("Migration and foreign-key checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
