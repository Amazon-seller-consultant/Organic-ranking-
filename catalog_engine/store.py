"""Multi-tenant persistence. One SQLite database, every row scoped by
seller_id; per-seller artifact directories under data/sellers/<seller_id>/.

Nothing here is seller-shared except the rules cache, which is keyed by
Amazon's templateIdentifier (the rules come from Amazon's template, not from
seller data, so sharing them leaks nothing between tenants).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
import dataclasses
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .exceptions import SellerNotFoundError
from .models import ComplianceLogEntry, SellerConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS sellers (
    seller_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    seller_id TEXT NOT NULL REFERENCES sellers(seller_id),
    source_file TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS sku_fingerprints (
    seller_id TEXT NOT NULL,
    sku TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (seller_id, sku)
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    sku TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rules_cache (
    template_identifier TEXT NOT NULL,
    product_type TEXT NOT NULL,
    rules_json TEXT NOT NULL,
    extracted_at REAL NOT NULL,
    PRIMARY KEY (template_identifier, product_type)
);
CREATE TABLE IF NOT EXISTS compliance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    sku TEXT NOT NULL,
    field_name TEXT NOT NULL,
    category TEXT NOT NULL,
    removed_text TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

RULES_CACHE_TTL_SECONDS = 14 * 24 * 3600  # refresh every 14 days by default


class Store:
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "engine.db"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- sellers ------------------------------------------------------------
    def upsert_seller(self, config: SellerConfig) -> None:
        self._conn.execute(
            "INSERT INTO sellers (seller_id, config_json, created_at) VALUES (?,?,?) "
            "ON CONFLICT(seller_id) DO UPDATE SET config_json=excluded.config_json",
            (config.seller_id, json.dumps(asdict(config)), time.time()),
        )
        self._conn.commit()
        self.seller_dir(config.seller_id)

    def get_seller(self, seller_id: str) -> SellerConfig:
        row = self._conn.execute(
            "SELECT config_json FROM sellers WHERE seller_id=?", (seller_id,)
        ).fetchone()
        if not row:
            raise SellerNotFoundError(
                f"seller '{seller_id}' is not registered; create it first"
            )
        data = json.loads(row[0])
        # tolerate config saved by a newer/older code version
        fields = {f.name for f in dataclasses.fields(SellerConfig)}
        return SellerConfig(**{k: v for k, v in data.items() if k in fields})

    def seller_dir(self, seller_id: str) -> Path:
        d = self.data_dir / "sellers" / seller_id
        (d / "uploads").mkdir(parents=True, exist_ok=True)
        (d / "outputs").mkdir(parents=True, exist_ok=True)
        return d

    # -- runs -----------------------------------------------------------------
    def start_run(self, seller_id: str, source_file: str) -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._conn.execute(
            "INSERT INTO runs (run_id, seller_id, source_file, started_at) VALUES (?,?,?,?)",
            (run_id, seller_id, source_file, time.time()),
        )
        self._conn.commit()
        return run_id

    def list_runs(self, seller_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, source_file, started_at, finished_at, summary_json "
            "FROM runs WHERE seller_id=? ORDER BY started_at DESC LIMIT ?",
            (seller_id, limit),
        ).fetchall()
        return [
            {
                "run_id": r[0], "source_file": r[1], "started_at": r[2],
                "finished_at": r[3],
                "summary": json.loads(r[4]) if r[4] else None,
            }
            for r in rows
        ]

    def finish_run(self, run_id: str, summary: dict[str, Any]) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at=?, summary_json=? WHERE run_id=?",
            (time.time(), json.dumps(summary, default=str), run_id),
        )
        self._conn.commit()

    # -- incremental fingerprints --------------------------------------------
    def get_fingerprints(self, seller_id: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT sku, fingerprint FROM sku_fingerprints WHERE seller_id=?",
            (seller_id,),
        ).fetchall()
        return dict(rows)

    def set_fingerprint(self, seller_id: str, sku: str, fingerprint: str) -> None:
        self._conn.execute(
            "INSERT INTO sku_fingerprints (seller_id, sku, fingerprint, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(seller_id, sku) DO UPDATE SET "
            "fingerprint=excluded.fingerprint, updated_at=excluded.updated_at",
            (seller_id, sku, fingerprint, time.time()),
        )
        self._conn.commit()

    # -- usage tracking --------------------------------------------------------
    def record_usage(
        self, seller_id: str, run_id: str, sku: str, model: str,
        input_tokens: int, output_tokens: int,
    ) -> None:
        self._conn.execute(
            "INSERT INTO usage (seller_id, run_id, sku, model, input_tokens, "
            "output_tokens, created_at) VALUES (?,?,?,?,?,?,?)",
            (seller_id, run_id, sku, model, input_tokens, output_tokens, time.time()),
        )
        self._conn.commit()

    def usage_totals(self, seller_id: str) -> dict[str, int]:
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) "
            "FROM usage WHERE seller_id=?",
            (seller_id,),
        ).fetchone()
        return {"calls": row[0], "input_tokens": row[1], "output_tokens": row[2]}

    # -- rules cache -----------------------------------------------------------
    def get_cached_rules(
        self, template_identifier: str, product_type: str,
        max_age_seconds: float = RULES_CACHE_TTL_SECONDS,
    ) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT rules_json, extracted_at FROM rules_cache "
            "WHERE template_identifier=? AND product_type=?",
            (template_identifier, product_type),
        ).fetchone()
        if not row:
            return None
        if time.time() - row[1] > max_age_seconds:
            return None  # stale — caller re-extracts from the current file
        return json.loads(row[0])

    def cache_rules(
        self, template_identifier: str, product_type: str, rules: dict[str, Any]
    ) -> None:
        self._conn.execute(
            "INSERT INTO rules_cache (template_identifier, product_type, rules_json, "
            "extracted_at) VALUES (?,?,?,?) ON CONFLICT(template_identifier, product_type) "
            "DO UPDATE SET rules_json=excluded.rules_json, extracted_at=excluded.extracted_at",
            (template_identifier, product_type, json.dumps(rules), time.time()),
        )
        self._conn.commit()

    # -- compliance audit log ----------------------------------------------------
    def log_compliance(self, entry: ComplianceLogEntry) -> None:
        self._conn.execute(
            "INSERT INTO compliance_log (seller_id, run_id, sku, field_name, category, "
            "removed_text, reason, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (entry.seller_id, entry.run_id, entry.sku, entry.field_name,
             entry.category, entry.removed_text, entry.reason, time.time()),
        )
        self._conn.commit()

    def compliance_entries(self, seller_id: str, run_id: str) -> list[tuple]:
        return self._conn.execute(
            "SELECT sku, field_name, category, removed_text, reason FROM compliance_log "
            "WHERE seller_id=? AND run_id=? ORDER BY id",
            (seller_id, run_id),
        ).fetchall()
