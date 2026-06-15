"""
Google Docs + Sheets + Slides integration for long-form documentation.

Uses a Google Service Account (server-to-server, no OAuth flow). Activates when
GOOGLE_SERVICE_ACCOUNT_JSON env var is set to the raw JSON of a service account
key, OR GOOGLE_SERVICE_ACCOUNT_FILE points to a JSON file path.

What it writes:
  • Daily standup minutes → appended to one rolling Google Doc per squad
  • Alpha review notes → one Doc per strategy, updated when reviewed
  • Weekly C-suite minutes → one rolling Doc per week
  • Daily P&L → appended row to one shared Sheet
  • Monthly board deck → generated as Slides from a template

Setup (one-time):
  1. https://console.cloud.google.com → create project "QuantEdge"
  2. APIs & Services → Library → enable: Docs API, Sheets API, Slides API, Drive API
  3. Credentials → Create credentials → Service account
  4. Download the JSON key
  5. Share each target Doc/Sheet/Slide with the service account email
     (looks like quantedge-bot@quantedge-12345.iam.gserviceaccount.com)
  6. Add env var GOOGLE_SERVICE_ACCOUNT_JSON to Render with the raw JSON contents

This module is intentionally dependency-light at import time: google-api-python-client
imports are inside functions so missing the dep doesn't break the app boot.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from app.utils.logging import logger

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive",
]


def _load_credentials():
    """Return google.oauth2.service_account.Credentials, or None if not configured."""
    try:
        from google.oauth2 import service_account
    except ImportError:
        logger.debug("google-auth not installed — google_docs disabled")
        return None

    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw_json:
        try:
            info = json.loads(raw_json)
            return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception as e:
            logger.warning("Invalid GOOGLE_SERVICE_ACCOUNT_JSON", error=str(e))
            return None

    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if file_path and os.path.exists(file_path):
        try:
            return service_account.Credentials.from_service_account_file(file_path, scopes=SCOPES)
        except Exception as e:
            logger.warning("Invalid GOOGLE_SERVICE_ACCOUNT_FILE", error=str(e))
    return None


def is_configured() -> bool:
    return _load_credentials() is not None


# ── Google Docs ────────────────────────────────────────────────────────────

def append_to_doc(doc_id: str, text: str, *, heading: str | None = None) -> bool:
    """Append a section to an existing Google Doc."""
    creds = _load_credentials()
    if not creds:
        return False
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return False

    try:
        service = build("docs", "v1", credentials=creds, cache_discovery=False)
        # Get end-of-doc index
        doc = service.documents().get(documentId=doc_id).execute()
        end_index = doc.get("body", {}).get("content", [])[-1].get("endIndex", 1) - 1

        requests = []
        if heading:
            requests.extend([
                {"insertText": {"location": {"index": end_index}, "text": f"\n{heading}\n"}},
                {"updateParagraphStyle": {
                    "range": {"startIndex": end_index + 1, "endIndex": end_index + 1 + len(heading) + 1},
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "fields": "namedStyleType",
                }},
            ])
            end_index += len(heading) + 2
        requests.append({"insertText": {"location": {"index": end_index}, "text": f"{text}\n"}})

        service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
        return True
    except Exception as e:
        logger.warning("google_docs.append_to_doc failed", error=str(e))
        return False


def create_doc(title: str, folder_id: str | None = None) -> str | None:
    """Create a new Google Doc, optionally in a folder. Returns the doc ID."""
    creds = _load_credentials()
    if not creds:
        return None
    try:
        from googleapiclient.discovery import build

        docs = build("docs", "v1", credentials=creds, cache_discovery=False)
        new_doc = docs.documents().create(body={"title": title}).execute()
        doc_id = new_doc["documentId"]

        if folder_id:
            drive = build("drive", "v3", credentials=creds, cache_discovery=False)
            drive.files().update(fileId=doc_id, addParents=folder_id,
                                 removeParents="root", fields="id, parents").execute()
        return doc_id
    except Exception as e:
        logger.warning("google_docs.create_doc failed", error=str(e))
        return None


# ── Google Sheets ──────────────────────────────────────────────────────────

def append_to_sheet(sheet_id: str, range_name: str, values: list[list[Any]]) -> bool:
    """Append rows to a Google Sheet range like 'PnL!A1'."""
    creds = _load_credentials()
    if not creds:
        return False
    try:
        from googleapiclient.discovery import build

        sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        sheets.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range=range_name,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": values},
        ).execute()
        return True
    except Exception as e:
        logger.warning("google_docs.append_to_sheet failed", error=str(e))
        return False


# ── High-level helpers ────────────────────────────────────────────────────

def log_standup(squad: str, shipped: list[str], planned: list[str], blockers: list[str]) -> bool:
    """Append today's standup to the squad's rolling minutes Doc."""
    doc_id = os.getenv(f"GDOC_STANDUP_{squad.upper().replace('-', '_')}", "").strip()
    if not doc_id:
        return False
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    body = (
        "Shipped:\n" + "\n".join(f"  • {x}" for x in shipped) + "\n\n"
        "Planned:\n" + "\n".join(f"  • {x}" for x in planned) + "\n\n"
        "Blockers:\n" + ("\n".join(f"  • {x}" for x in blockers) if blockers else "  (none)")
    )
    return append_to_doc(doc_id, body, heading=f"Standup {today}")


def log_pnl_row(date: str, strategy: str, pnl: float, win_rate: float, slippage_bps: float) -> bool:
    """Append one row to the master P&L sheet."""
    sheet_id = os.getenv("GSHEET_PNL_DAILY", "").strip()
    if not sheet_id:
        return False
    return append_to_sheet(sheet_id, "Daily!A:E", [[date, strategy, pnl, win_rate, slippage_bps]])


def log_alpha_review(strategy: str, sharpe: float, maxdd: float, decision: str, notes: str) -> bool:
    """Append an alpha review to the rolling alpha-research Doc."""
    doc_id = os.getenv("GDOC_ALPHA_REVIEWS", "").strip()
    if not doc_id:
        return False
    body = (
        f"Sharpe: {sharpe:.2f}\n"
        f"Max drawdown: {maxdd:.1%}\n"
        f"Decision: {decision}\n\n"
        f"Notes:\n{notes}"
    )
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return append_to_doc(doc_id, body, heading=f"{today} — {strategy}")
