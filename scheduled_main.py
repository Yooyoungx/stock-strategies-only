"""
GitHub Actions 專用入口。

手動 Run workflow：
    一律執行，方便測試。

schedule：
    先檢查 Google Sheet RunLog。
    今天若已成功發送，備援排程直接跳過；
    否則執行 main.py，成功後才記錄 SENT。
"""

import os

import main as signal_main
from stock_strategies.run_guard import (
    already_sent_today,
    mark_sent_today,
    taiwan_today,
)


def run():
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    is_scheduled = event_name == "schedule"

    if is_scheduled:
        today = taiwan_today()

        print(f"🕒 排程執行日期（台灣）：{today}")
        print("🔎 檢查今天是否已成功發送 Telegram...")

        if already_sent_today():
            print("✅ 今天已成功發送，這次是備援排程，不重複執行。")
            return

        print("➡️ 今天尚未成功發送，開始執行每日選股。")
    else:
        print("🧪 手動 workflow_dispatch：略過防重複檢查，直接執行。")

    # 若 main.main() 中途丟出例外 / sys.exit(1)，
    # 程式不會走到 mark_sent_today()，備援排程仍會補跑。
    signal_main.main()

    if is_scheduled:
        mark_sent_today()
        print("✅ 已在 Google Sheet RunLog 記錄 SENT。")


if __name__ == "__main__":
    run()
