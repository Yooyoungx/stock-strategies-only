"""
短線動能 + 箱型策略 V1

核心邏輯：
1. 成交量至少 7000 張
2. 近期強勢：20 日 +10% 或 60 日 +20%
3. 中長期多頭：MA60 > MA120，且 MA60 持續上升
4. 箱型寬度 <= 20%
5. 兩種進場訊號：
   A. BOX_BOTTOM：強勢股回到箱底附近，量縮止穩
   B. BOX_BREAKOUT：放量突破箱頂
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from .config import CONFIG
from .data import get_price_history
from .volume import detect_patterns, verdict as volume_verdict


# ============================================================
# 可以自行調整的策略參數
# ============================================================

# FinMind 的 volume 是「股」，7000 張 = 7,000,000 股
MIN_VOLUME_SHARES = 7_000_000

# 動能條件
MIN_20D_RETURN = 0.10       # 20 日至少 +10%
MIN_60D_RETURN = 0.20       # 或 60 日至少 +20%

# 箱型
BOX_DAYS = 20               # 先用最近 20 個交易日判定箱型
MAX_BOX_WIDTH = 0.20        # 箱型高低差最多 20%
BOX_TOUCH_TOLERANCE = 0.03  # 距箱頂/箱底 3% 內視為碰觸
MIN_BOX_TOUCHES = 2         # 箱頂、箱底至少各碰 2 次

# 箱底買點
BOTTOM_DISTANCE = 0.05      # 距離箱底 5% 內
BOTTOM_VOLUME_RATIO = 0.90  # 5 日均量 <= 20 日均量 90%

# 突破買點
BREAKOUT_BUFFER = 0.01      # 收盤至少突破箱頂 1%
BREAKOUT_VOLUME_RATIO = 1.50  # 當日量至少為 20 日均量 1.5 倍

# 排除產業
EXCLUDED_CATEGORY_KEYWORDS = (
    "金融",
    "銀行",
    "保險",
    "證券",
    "生技",
    "生技醫療",
    "醫療",
)


def evaluate_momentum_box(
    stock_id: str,
    name: str,
    category: str = "",
) -> Optional[dict]:
    """評估單一股票的動能 + 箱型策略。"""

    result = {
        "stock_id": str(stock_id),
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strategy_id": "momentum_box",
        "risk_notes": [],
    }

    try:
        # ----------------------------------------------------
        # 0. 排除金融 / 生技
        # ----------------------------------------------------
        category_text = str(category or "")

        if any(keyword in category_text for keyword in EXCLUDED_CATEGORY_KEYWORDS):
            result.update({
                "action": "SKIP",
                "signal_score": 0,
                "components": {
                    "fundamental_pass": None,
                    "tech_score": 0,
                    "tech_signals": [],
                    "backtest_winrate": None,
                    "backtest_samples": 0,
                    "volume_patterns": [],
                    "volume_details": {},
                    "volume_bonus": 0,
                    "volume_verdict": "",
                },
                "trend": {},
            })

            result["risk_notes"].append(
                f"排除產業：{category_text}"
            )

            return result

        # ----------------------------------------------------
        # 1. 抓價格資料
        # ----------------------------------------------------
        px = get_price_history(stock_id, years=1)

        if px.empty or len(px) < 140:
            result["action"] = "SKIP"
            result["signal_score"] = 0
            result["risk_notes"].append("價格資料不足 140 個交易日")
            return result

        px = px.copy()

        # ----------------------------------------------------
        # 2. 均線
        # ----------------------------------------------------
        px["ma20"] = px["close"].rolling(20).mean()
        px["ma60"] = px["close"].rolling(60).mean()
        px["ma120"] = px["close"].rolling(120).mean()

        latest = px.iloc[-1]
        prev = px.iloc[-2]

        close = float(latest["close"])
        open_price = float(latest["open"])
        volume = float(latest["volume"])

        ma20 = float(latest["ma20"])
        ma60 = float(latest["ma60"])
        ma120 = float(latest["ma120"])

        # ----------------------------------------------------
        # 3. 最近漲幅
        # ----------------------------------------------------
        chg_5d = close / float(px.iloc[-6]["close"]) - 1
        chg_20d = close / float(px.iloc[-21]["close"]) - 1
        chg_60d = close / float(px.iloc[-61]["close"]) - 1

        # ----------------------------------------------------
        # 4. 成交量
        # ----------------------------------------------------
        # 用前 20 日均量，不包含今天
        vol_20 = float(px["volume"].iloc[-21:-1].mean())
        vol_5 = float(px["volume"].iloc[-5:].mean())

        vol_ratio = volume / vol_20 if vol_20 > 0 else 0

        liquidity_ok = volume >= MIN_VOLUME_SHARES

        # ----------------------------------------------------
        # 5. 動能判斷
        # ----------------------------------------------------
        momentum_ok = (
            chg_20d >= MIN_20D_RETURN
            or chg_60d >= MIN_60D_RETURN
        )

        # ----------------------------------------------------
        # 6. 中長期多頭趨勢
        # ----------------------------------------------------
        ma60_20days_ago = float(px.iloc[-21]["ma60"])

        ma60_rising = ma60 > ma60_20days_ago

        long_trend_ok = (
            close > ma120
            and ma60 > ma120
            and ma60_rising
        )

        strong_alignment = (
            close > ma20 > ma60 > ma120
        )

        # ----------------------------------------------------
        # 7. 箱型判定
        #
        # 重要：
        # 箱型使用「今天以前」20 日，
        # 不讓今天突破的價格把箱頂一起抬高。
        # ----------------------------------------------------
        box = px.iloc[-(BOX_DAYS + 1):-1]

        box_high = float(box["high"].max())
        box_low = float(box["low"].min())

        if box_low <= 0:
            return None

        box_width = (box_high - box_low) / box_low

        high_touches = int(
            (
                box["high"]
                >= box_high * (1 - BOX_TOUCH_TOLERANCE)
            ).sum()
        )

        low_touches = int(
            (
                box["low"]
                <= box_low * (1 + BOX_TOUCH_TOLERANCE)
            ).sum()
        )

        box_valid = (
            box_width <= MAX_BOX_WIDTH
            and high_touches >= MIN_BOX_TOUCHES
            and low_touches >= MIN_BOX_TOUCHES
        )

        # ----------------------------------------------------
        # 8. 箱底買點
        # ----------------------------------------------------
        distance_to_bottom = close / box_low - 1

        near_bottom = (
            0 <= distance_to_bottom <= BOTTOM_DISTANCE
        )

        pullback_volume_ok = (
            vol_20 > 0
            and vol_5 <= vol_20 * BOTTOM_VOLUME_RATIO
        )

        # 不希望一路下跌就直接接刀
        stabilizing = (
            close >= open_price
            or close >= float(prev["close"])
        )

        bottom_signal = (
            liquidity_ok
            and momentum_ok
            and long_trend_ok
            and box_valid
            and near_bottom
            and pullback_volume_ok
            and stabilizing
        )

        # ----------------------------------------------------
        # 9. 箱型突破
        # ----------------------------------------------------
        breakout_price_ok = (
            close >= box_high * (1 + BREAKOUT_BUFFER)
        )

        breakout_volume_ok = (
            vol_ratio >= BREAKOUT_VOLUME_RATIO
        )

        breakout_signal = (
            liquidity_ok
            and momentum_ok
            and long_trend_ok
            and box_valid
            and breakout_price_ok
            and breakout_volume_ok
            and close > ma20
        )

        # ----------------------------------------------------
        # 10. 評分
        # ----------------------------------------------------
        score = 0
        signals = []

        if liquidity_ok:
            score += 15
            signals.append("成交量≥7000張")

        if momentum_ok:
            score += 20
            signals.append("近期強勢動能")

        if chg_20d >= 0.20 or chg_60d >= 0.30:
            score += 10
            signals.append("高動能")

        if long_trend_ok:
            score += 20
            signals.append("中長期多頭")

        if strong_alignment:
            score += 10
            signals.append("均線多頭排列")

        if box_valid:
            score += 15
            signals.append("箱型成立")

        setup = "NONE"

        if breakout_signal:
            score += 25
            setup = "BOX_BREAKOUT"
            signals.append("箱型放量突破")

        elif bottom_signal:
            score += 20
            setup = "BOX_BOTTOM"
            signals.append("箱底量縮止穩")

        score = min(100, score)

        # ----------------------------------------------------
        # 11. BUY / WATCH / SKIP
        # ----------------------------------------------------
        if breakout_signal:
            action = "BUY"

        elif bottom_signal:
            action = "BUY"

        elif (
            liquidity_ok
            and momentum_ok
            and long_trend_ok
        ):
            action = "WATCH"
            setup = "HOT"

        else:
            action = "SKIP"

        # ----------------------------------------------------
        # 12. 風險原因
        # ----------------------------------------------------
        if not liquidity_ok:
            result["risk_notes"].append(
                f"成交量不足 7000 張，目前約 {volume / 1000:,.0f} 張"
            )

        if not momentum_ok:
            result["risk_notes"].append(
                "近期動能不足"
            )

        if not long_trend_ok:
            result["risk_notes"].append(
                "中長期趨勢尚未符合 MA60 > MA120 且 MA60 向上"
            )

        if (
            liquidity_ok
            and momentum_ok
            and long_trend_ok
            and not box_valid
        ):
            result["risk_notes"].append(
                "強勢股，但尚未形成穩定箱型"
            )

        if action == "WATCH" and setup == "HOT":
            result["risk_notes"].append(
                "強勢股，等待箱底或箱型突破買點"
            )

        # ----------------------------------------------------
        # 13. 原專案量價分析保留
        # ----------------------------------------------------
        vp = detect_patterns(px)

        # ----------------------------------------------------
        # 14. 停損 / 停利
        #
        # V1 暫時沿用原專案 CONFIG，
        # 這樣 Performance 成績單仍能一致追蹤。
        # ----------------------------------------------------
        stop_loss = CONFIG["stop_loss"]
        target_return = CONFIG["target_return"]

        entry_price = close
        stop_price = round(
            entry_price * (1 - stop_loss),
            2,
        )
        target_price = round(
            entry_price * (1 + target_return),
            2,
        )

        rr = round(
            target_return / stop_loss,
            2,
        )

        position_pct = min(
            2.0 / (stop_loss * 100) * 100,
            20.0,
        )

        # 年度最高價
        high_252 = float(
            px["high"].iloc[-252:].max()
            if len(px) >= 252
            else px["high"].max()
        )

        pct_from_high = (
            close / high_252 - 1
        ) * 100

        result.update({
            "action": action,
            "signal_score": score,

            "components": {
                # 此策略不使用 EPS / ROE
                "fundamental_pass": None,

                "tech_score": score,
                "tech_signals": signals,

                # V1 尚未替箱型策略重寫歷史回測
                "backtest_winrate": None,
                "backtest_samples": 0,

                "volume_patterns": vp["patterns"],
                "volume_details": vp["details"],
                "volume_bonus": vp["bonus"],
                "volume_verdict": volume_verdict(
                    vp["patterns"]
                ),
            },

            "trend": {
                "setup": setup,

                "chg_5d": round(
                    chg_5d * 100,
                    2,
                ),
                "chg_20d": round(
                    chg_20d * 100,
                    2,
                ),
                "chg_60d": round(
                    chg_60d * 100,
                    2,
                ),

                "vol_ratio": round(
                    vol_ratio,
                    2,
                ),
                "volume_lots": round(
                    volume / 1000,
                    0,
                ),

                "pct_from_high": round(
                    pct_from_high,
                    1,
                ),

                "above_ma20": bool(
                    close > ma20
                ),
                "above_ma60": bool(
                    close > ma60
                ),

                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "ma120": round(ma120, 2),

                "box_high": round(
                    box_high,
                    2,
                ),
                "box_low": round(
                    box_low,
                    2,
                ),
                "box_width_pct": round(
                    box_width * 100,
                    1,
                ),

                "box_high_touches": high_touches,
                "box_low_touches": low_touches,

                "distance_to_bottom_pct": round(
                    distance_to_bottom * 100,
                    1,
                ),
            },

            "entry_price": round(
                entry_price,
                2,
            ),
            "stop_loss_price": stop_price,
            "target_price": target_price,
            "risk_reward_ratio": rr,
            "position_size_pct": round(
                position_pct,
                1,
            ),

            "entry_rule": (
                "動能箱型策略："
                "箱底量縮止穩或放量突破箱頂"
            ),
        })

        return result

    except Exception as e:
        result["action"] = "ERROR"
        result["signal_score"] = 0
        result["risk_notes"].append(
            f"錯誤: {str(e)[:120]}"
        )
        return result


# ============================================================
# Telegram 專用格式
# ============================================================

def _setup_name(setup: str) -> str:
    return {
        "BOX_BREAKOUT": "🚀 箱型突破",
        "BOX_BOTTOM": "🎯 箱底伏擊",
        "HOT": "🔥 強勢觀察",
        "NONE": "—",
    }.get(setup, setup)


def _format_one_stock(s: dict) -> list[str]:
    t = s.get("trend", {})
    c = s.get("components", {})

    action = s.get("action", "")
    icon = {
        "BUY": "🟢",
        "WATCH": "🟡",
        "SKIP": "⚪",
        "ERROR": "🔴",
    }.get(action, "⚪")

    setup = _setup_name(
        t.get("setup", "NONE")
    )

    lines = [
        (
            f"{icon} *{s['stock_id']} {s['name']}* "
            f"| {setup} | {s.get('signal_score', 0)} 分"
        )
    ]

    if t:
        lines.append(
            f"現價 {s.get('entry_price')} | "
            f"箱頂 {t.get('box_high')} | "
            f"箱底 {t.get('box_low')} | "
            f"箱寬 {t.get('box_width_pct')}%"
        )

        lines.append(
            f"20日 {t.get('chg_20d', 0):+.1f}% | "
            f"60日 {t.get('chg_60d', 0):+.1f}% | "
            f"量 {t.get('volume_lots', 0):,.0f} 張 | "
            f"量比 {t.get('vol_ratio', 0):.2f}x"
        )

    tech_signals = c.get(
        "tech_signals",
        [],
    )

    if tech_signals:
        lines.append(
            "訊號: "
            + " / ".join(tech_signals)
        )

    if action == "BUY":
        lines.append(
            f"參考風控：停損 {s.get('stop_loss_price')} | "
            f"目標 {s.get('target_price')}"
        )

    if s.get("risk_notes"):
        lines.append(
            "⚠️ "
            + " / ".join(
                s["risk_notes"]
            )
        )

    return lines


def format_momentum_messages(
    signals: list[dict],
    market: dict | None = None,
    night_note: str | None = None,
) -> list[str]:

    buys = [
        s for s in signals
        if s.get("action") == "BUY"
    ]

    watches = [
        s for s in signals
        if s.get("action") == "WATCH"
    ]

    skips = [
        s for s in signals
        if s.get("action") in (
            "SKIP",
            "ERROR",
        )
    ]

    today = datetime.now().strftime(
        "%Y/%m/%d"
    )

    # -----------------------------------------
    # 第一則
    # -----------------------------------------
    msg1 = [
        f"🔥 *動能箱型選股 V1* {today}",
        (
            f"掃描 {len(signals)} 檔 | "
            f"BUY {len(buys)} | "
            f"WATCH {len(watches)} | "
            f"SKIP {len(skips)}"
        ),
        "",
    ]

    if market and market.get("note"):
        msg1.extend([
            "🎯 *大盤濾鏡*",
            market["note"],
            "",
        ])

    if night_note:
        msg1.extend([
            "🌙 *夜盤濾鏡*",
            night_note,
            "",
        ])

    msg1.extend([
        "📋 *策略條件*",
        "• 成交量 ≥ 7000 張",
        "• 20日漲幅 ≥10% 或 60日 ≥20%",
        "• MA60 > MA120 且 MA60 向上",
        "• 20日箱型寬度 ≤20%",
        "• 箱底：距箱底 ≤5% + 量縮止穩",
        "• 突破：收盤突破箱頂 ≥1% + 量比 ≥1.5x",
    ])

    # -----------------------------------------
    # 第二則
    # -----------------------------------------
    msg2 = []

    if buys:
        msg2.extend([
            f"🟢 *今日動能箱型訊號 ({len(buys)})*",
            "",
        ])

        for s in buys[:10]:
            msg2.extend(
                _format_one_stock(s)
            )
            msg2.append("")

    else:
        msg2.extend([
            "🟢 *今日沒有箱底 / 箱型突破訊號*",
            "",
        ])

    if watches:
        msg2.extend([
            f"🔥 *強勢觀察 TOP {min(10, len(watches))}*",
            "",
        ])

        for s in watches[:10]:
            msg2.extend(
                _format_one_stock(s)
            )
            msg2.append("")

    # -----------------------------------------
    # 第三則
    # -----------------------------------------
    msg3 = [
        "🧠 *策略說明*",
        "",
        "BUY = 已出現箱底止穩或箱型突破條件",
        "WATCH = 動能與長期趨勢符合，但買點尚未出現",
        "SKIP = 流動性 / 動能 / 趨勢條件未達標",
        "",
        "_系統訊號僅供研究與紀錄，不代表保證獲利。_",
    ]

    return [
        "\n".join(msg1),
        "\n".join(msg2),
        "\n".join(msg3),
    ]
