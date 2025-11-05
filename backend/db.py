# backend/db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("fmea.sqlite3")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def migrate():
    schema = r"""
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS ontology_class (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT CHECK(type IN ('Product','Process','Resource')) NOT NULL,
    parent_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL,
    description TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_ontology_name_type ON ontology_class(name, type);

    CREATE TABLE IF NOT EXISTS risk_failure_mode (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    process_class_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS risk_cause (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    resource_class_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS risk_effect (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    product_class_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS risk_control (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT
    );

    CREATE TABLE IF NOT EXISTS failure_cause (
    failure_mode_id INTEGER REFERENCES risk_failure_mode(id) ON DELETE CASCADE,
    cause_id INTEGER REFERENCES risk_cause(id) ON DELETE CASCADE,
    PRIMARY KEY (failure_mode_id, cause_id)
    );

    CREATE TABLE IF NOT EXISTS failure_effect (
    failure_mode_id INTEGER REFERENCES risk_failure_mode(id) ON DELETE CASCADE,
    effect_id INTEGER REFERENCES risk_effect(id) ON DELETE CASCADE,
    PRIMARY KEY (failure_mode_id, effect_id)
    );

    CREATE TABLE IF NOT EXISTS failure_control (
    failure_mode_id INTEGER REFERENCES risk_failure_mode(id) ON DELETE CASCADE,
    control_id INTEGER REFERENCES risk_control(id) ON DELETE CASCADE,
    PRIMARY KEY (failure_mode_id, control_id)
    );

    CREATE TABLE IF NOT EXISTS case_header (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS case_ppr_binding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER REFERENCES case_header(id) ON DELETE CASCADE,
    product_class_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL,
    process_class_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL,
    resource_class_id INTEGER REFERENCES ontology_class(id) ON DELETE SET NULL
    );

   CREATE TABLE IF NOT EXISTS case_fmea (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER REFERENCES case_header(id) ON DELETE CASCADE,
    system_element TEXT,            -- NEW: added for grouping
    failure_mode_id INTEGER REFERENCES risk_failure_mode(id) ON DELETE SET NULL,
    cause_id INTEGER REFERENCES risk_cause(id) ON DELETE SET NULL,
    effect_id INTEGER REFERENCES risk_effect(id) ON DELETE SET NULL,
    control_id INTEGER REFERENCES risk_control(id) ON DELETE SET NULL,
    severity INTEGER,
    occurrence INTEGER,
    detection INTEGER,
    rpn INTEGER GENERATED ALWAYS AS (COALESCE(severity,0) * COALESCE(occurrence,0) * COALESCE(detection,0)) VIRTUAL,
    notes TEXT
);


    CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    vector BLOB NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_table_row ON embeddings(table_name, row_id);
    """
    with get_conn() as conn:
        conn.executescript(schema)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"[DB] Tables after migrate (inline): {[t for t in tables]}")

def init_and_seed():
    migrate()
    from .seed import seed
    seed()
