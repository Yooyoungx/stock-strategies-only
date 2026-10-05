"""
GitHub Actions 每日選股排程防重複機制。

用途：
- 主排程成功發送 Telegram 後，在 Google Sheet 的 RunLog 分頁記錄 SENT。
- 備援排程啟動時，如果今天已經有 SENT，就直接跳過。
- 如果主排程沒觸發或執行失敗，因為不會寫 SENT，所以備援會正常補跑。
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread

from .sheet import get_gsheet


TAIPEI_TZ = ZoneInfo("Asia/Taipei")
WORKSHEET_NAME = "RunLog"
WORKFLOW_NAME = "daily_signal"

HEADERS = [
    "date",
    "workflow",
    "status",
    "sent_at",
    "github_run_id",
]


def taiwan_now() -> datetime:
    return datetime.now(TAIPEI_TZ)


def taiwan_today() -> str:
    return taiwan_now().strftime("%Y-%m-%d")


def _get_or_create_runlog():
    sh = get_gsheet()

    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=WORKSHEET_NAME,
            rows=500,
            cols=len(HEADERS),
        )
        ws.append_row(HEADERS)

    return ws


def already_sent_today() -> bool:
    """
    今天是否已經成功完成每日選股 Telegram 發送。

    RunLog 不存在時視為尚未發送。
    """
    sh = get_gsheet()

    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        return False

    today = taiwan_today()
    rows = ws.get_all_records()

    # 從最新紀錄往回找，通常只需要看很少幾列。
    for row in reversed(rows):
        row_date = str(row.get("date", "")).strip()

        if row_date and row_date < today:
            break

        if (
            row_date == today
            and str(row.get("workflow", "")).strip() == WORKFLOW_NAME
            and str(row.get("status", "")).strip().upper() == "SENT"
        ):
            return True

    return False


def mark_sent_today():
    """
    只有 main.py 完整成功結束後才呼叫。
    """
    ws = _get_or_create_runlog()

    now = taiwan_now()
    ws.append_row([
        now.strftime("%Y-%m-%d"),
        WORKFLOW_NAME,
        "SENT",
        now.strftime("%Y-%m-%d %H:%M:%S"),
        os.environ.get("GITHUB_RUN_ID", ""),
    ])
