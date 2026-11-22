"""Human-in-the-loop SQLite review queue for borderline safety scores."""
import sqlite3
import csv
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

class ReviewQueue:
    def __init__(self, db_path: str = "review_queue.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    margin REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    human_label INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP
                )
            """)
            conn.commit()

    def maybe_enqueue(self, text: str, score: float, threshold: float, margin: float, model_version: str) -> Optional[int]:
        """Enqueues if |score - threshold| < margin."""
        if abs(score - threshold) < margin:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO review_items (text, score, threshold, margin, model_version, status)
                    VALUES (?, ?, ?, ?, ?, 'pending')
                """, (text, float(score), float(threshold), float(margin), model_version))
                conn.commit()
                return cur.lastrowid
        return None

    def store_review(self, item_id: int, human_label: int) -> bool:
        """Stores human label (0=safe, 1=toxic/violating)."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE review_items
                SET human_label = ?, status = 'reviewed', reviewed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (int(human_label), int(item_id)))
            conn.commit()
            return cur.rowcount > 0

    def get_pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM review_items WHERE status = 'pending' ORDER BY id ASC LIMIT ?", (limit,))
            return [dict(row) for row in cur.fetchall()]

    def export_to_csv(self, csv_path: str) -> int:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, text, score, threshold, model_version, human_label, reviewed_at FROM review_items WHERE status = 'reviewed'")
            rows = cur.fetchall()
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "text", "score", "threshold", "model_version", "human_label", "reviewed_at"])
                for r in rows:
                    writer.writerow([r["id"], r["text"], r["score"], r["threshold"], r["model_version"], r["human_label"], r["reviewed_at"]])
            return len(rows)
