# BeatSafe - Cardiac Triage AI for Offline Primary Care Units in Brazil
# Powered by Gemma 3 (4B) via Ollama - runs fully offline
# Designed for Gemma 4 migration when available on Ollama
# Author: NamiShima
# Competition: Gemma 4 Good Hackathon 2026
import sqlite3                  # Built-in database — no install needed
import os                       # File path operations
from datetime import datetime   # Timestamps for each triage record

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP — SQLite file stored in the BeatSafe root folder
# Offline-first: no server, no internet, no configuration needed
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "beatsafe_history.db")


def init_db():
    """
    Creates the triagens table if it doesn't exist yet.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    Called automatically on module import.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triagens (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora     TEXT NOT NULL,
                patient_name  TEXT,
                age           INTEGER,
                sex           TEXT,
                chief_complaint TEXT,
                symptoms      TEXT,
                vitals        TEXT,
                history       TEXT,
                medications   TEXT,
                risk_level    TEXT,
                triage_result TEXT,
                ecg_analysis  TEXT,
                pdf_path      TEXT
            )
        """)
        conn.commit()


def save_triage(
    patient_name: str,
    age: int,
    sex: str,
    chief_complaint: str,
    symptoms: str,
    vitals: str,
    history: str,
    medications: str,
    triage_result: str,
    ecg_analysis: str = None,
    pdf_path: str = None
) -> int:
    """
    Saves a completed triage to the SQLite database.
    Automatically extracts the risk level from the triage result text.

    Args:
        patient_name:    Patient name (may be empty)
        age:             Patient age
        sex:             Biological sex
        chief_complaint: Main complaint
        symptoms:        Associated symptoms
        vitals:          Vital signs
        history:         Medical history
        medications:     Current medications
        triage_result:   Full output from Gemma 3
        ecg_analysis:    ECG analysis output (optional)
        pdf_path:        Path to the generated PDF (optional)

    Returns:
        ID of the inserted record
    """

    # Extract risk level from triage text for easy filtering
    result_upper = triage_result.upper()
    if "ALTO RISCO" in result_upper:
        risk_level = "🔴 Alto Risco"
    elif "MODERADO" in result_upper:
        risk_level = "🟡 Moderado"
    else:
        risk_level = "🟢 Baixo Risco"

    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M"  )  # ex: 16/05/2026 14:09

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO triagens (
                data_hora, patient_name, age, sex,
                chief_complaint, symptoms, vitals, history, medications,
                risk_level, triage_result, ecg_analysis, pdf_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data_hora, patient_name or "Não informado", age, sex,
            chief_complaint, symptoms, vitals, history, medications,
            risk_level, triage_result, ecg_analysis, pdf_path
        ))
        conn.commit()
        return cursor.lastrowid


def get_history(risk_filter: str = "Todos", limit: int = 50) -> list[list]:
    """
    Retrieves triage history from the database for display in Gradio.
    Returns rows formatted for gr.Dataframe.

    Args:
        risk_filter: "Todos", "🔴 Alto Risco", "🟡 Moderado", or "🟢 Baixo Risco"
        limit:       Maximum number of records to return (most recent first)

    Returns:
        List of rows: [data_hora, patient_name, age, risk_level, chief_complaint, vitals]
    """

    with sqlite3.connect(DB_PATH) as conn:
        if risk_filter == "Todos":
            cursor = conn.execute("""
                SELECT data_hora, patient_name, age, risk_level, chief_complaint, vitals
                FROM triagens
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor = conn.execute("""
                SELECT data_hora, patient_name, age, risk_level, chief_complaint, vitals
                FROM triagens
                WHERE risk_level = ?
                ORDER BY id DESC
                LIMIT ?
            """, (risk_filter, limit))

        return cursor.fetchall()


def get_triage_detail(record_id: int) -> dict:
    """
    Retrieves the full details of a single triage record.

    Args:
        record_id: The ID of the triage record

    Returns:
        Dictionary with all fields, or empty dict if not found
    """

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM triagens WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        return dict(row) if row else {}


def get_stats() -> dict:
    """
    Returns summary statistics for the history dashboard.

    Returns:
        Dict with total, alto_risco, moderado, baixo_risco counts
    """

    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM triagens").fetchone()[0]
        alto  = conn.execute("SELECT COUNT(*) FROM triagens WHERE risk_level = '🔴 Alto Risco'").fetchone()[0]
        mod   = conn.execute("SELECT COUNT(*) FROM triagens WHERE risk_level = '🟡 Moderado'").fetchone()[0]
        baixo = conn.execute("SELECT COUNT(*) FROM triagens WHERE risk_level = '🟢 Baixo Risco'").fetchone()[0]

    return {
        "total":      total,
        "alto_risco": alto,
        "moderado":   mod,
        "baixo_risco": baixo
    }


# Initialize database on import — safe and idempotent
init_db()
