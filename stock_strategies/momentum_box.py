"""
動能 + 箱型 + 多重背離策略 V2

核心：
1. 成交量至少 7000 張
2. 近期強勢：20 日 +10% 或 60 日 +20%
3. 中長期多頭：MA60 > MA120 且 MA60 向上
4. 20 日箱型：寬度 <= 20%、箱頂/箱底有足夠碰觸
5. 買點：箱底量縮止穩 / 放量突破箱頂
6. 背離確認：RSI / MFI / MACD / OBV
7. 背離歷史統計：TP 2.5 ATR、SL 1.5 ATR，顯示勝率 / CUM ATR / 期望值
"""

from datetime import datetime
from typing import Optional

from .config import CONFIG
from .data import get_price_history
from .divergence import analyze_divergence
from .volume import detect_patterns, verdict as volume_verdict


# ============================================================
# 策略參數
# ============================================================
MIN_VOLUME_SHARES = 7_000_000

MIN_20D_RETURN = 0.10
MIN_60D_RETURN = 0.20

BOX_DAYS = 20
MAX_BOX_WIDTH = 0.20
BOX_TOUCH_TOLERANCE = 0.03
MIN_BOX_TOUCHES = 2

BOTTOM_DISTANCE = 0.05
BOTTOM_VOLUME_RATIO = 0.90

BREAKOUT_BUFFER = 0.01
BREAKOUT_VOLUME_RATIO = 1.50

EXCLUDED_CATEGORY_KEYWORDS = (
    "金融",
    "銀行",
    "保險",
    "證券",
    "生技",
    "生技醫療",
    "醫療",
)


def _momentum_score(
    volume: float,
    vol_ratio: float,
    chg_20d: float,
    chg_60d: float,
    close: float,
    ma60: float,
    ma120: float,
    ma60_rising: bool,
) -> int:
    """0~100，刻意把近期漲幅與中長期趨勢拆開。"""
    score = 0

    # 流動性 / 熱度：20
    if volume >= MIN_VOLUME_SHARES:
        score += 15
    if vol_ratio >= 1.50:
        score += 5

    # 20 日動能：25
    if chg_20d >= 0.20:
        score += 25
    elif chg_20d >= MIN_20D_RETURN:
        score += 15

    # 60 日動能：25
    if chg_60d >= 0.30:
        score += 25
    elif chg_60d >= MIN_60D_RETURN:
        score += 15

    # 趨勢：30
    if close > ma120:
        score += 10
    if ma60 > ma120:
        score += 10
    if ma60_rising:
        score += 10

    return min(100, score)


def _box_score(box, box_width: float, high_touches: int, low_touches: int) -> int:
    """0~100：越窄、上下緣反覆確認、區間淨漂移越小，箱型品質越高。"""
    score = 0

    # 寬度：最多 50
    if box_width <= 0.10:
        score += 50
    elif box_width <= 0.15:
        score += 40
    elif box_width <= 0.20:
        score += 30
    elif box_width <= 0.25:
        score += 15

    # 觸碰次數：箱頂/箱底各最多 20
    if high_touches >= 3:
        score += 20
    elif high_touches >= 2:
        score += 10

    if low_touches >= 3:
        score += 20
    elif low_touches >= 2:
        score += 10

    # 箱內淨漂移：最多 10
    if len(box) >= 2:
        first_close = float(box.iloc[0]["close"])
        last_close = float(box.iloc[-1]["close"])
        drift = abs(last_close / first_close - 1) if first_close > 0 else 1.0
        if drift <= 0.05:
            score += 10
        elif drift <= 0.10:
            score += 5

    return min(100, score)


def _box_quality(score: int) -> str:
    if score >= 80:
        return "優秀"
    if score >= 65:
        return "良好"
    if score >= 50:
        return "普通"
    return "偏弱"


def evaluate_momentum_box(
    stock_id: str,
    name: str,
    category: str = "",
) -> Optional[dict]:
    """評估單一股票：動能 + 箱型 + 背離確認。"""

    result = {
        "stock_id": str(stock_id),
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strategy_id": "momentum_box_v2",
        "risk_notes": [],
    }

    try:
        # 0. 排除產業
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
                    "momentum_score": 0,
                    "box_score": 0,
                    "divergence_score": 50,
                    "volume_patterns": [],
                    "volume_details": {},
                    "volume_bonus": 0,
                    "volume_verdict": "",
                },
                "trend": {},
                "divergence": {},
            })
            result["risk_notes"].append(f"排除產業：{category_text}")
            return result

        # 1. 抓 3 年資料：最新選股 + 背離歷史勝率都用同一批資料
        px = get_price_history(stock_id, years=3)
        if px.empty or len(px) < 180:
            result["action"] = "SKIP"
            result["signal_score"] = 0
            result["risk_notes"].append("價格資料不足 180 個交易日")
            return result

        px = px.copy().reset_index(drop=True)
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

        # 2. 動能
        chg_5d = close / float(px.iloc[-6]["close"]) - 1
        chg_20d = close / float(px.iloc[-21]["close"]) - 1
        chg_60d = close / float(px.iloc[-61]["close"]) - 1

        vol_20 = float(px["volume"].iloc[-21:-1].mean())
        vol_5 = float(px["volume"].iloc[-5:].mean())
        vol_ratio = volume / vol_20 if vol_20 > 0 else 0.0

        liquidity_ok = volume >= MIN_VOLUME_SHARES
        momentum_ok = chg_20d >= MIN_20D_RETURN or chg_60d >= MIN_60D_RETURN

        ma60_20days_ago = float(px.iloc[-21]["ma60"])
        ma60_rising = ma60 > ma60_20days_ago
        long_trend_ok = close > ma120 and ma60 > ma120 and ma60_rising
        strong_alignment = close > ma20 > ma60 > ma120

        momentum_score = _momentum_score(
            volume=volume,
            vol_ratio=vol_ratio,
            chg_20d=chg_20d,
            chg_60d=chg_60d,
            close=close,
            ma60=ma60,
            ma120=ma120,
            ma60_rising=ma60_rising,
        )

        # 3. 箱型：今天以前 20 日，不把今日突破價納入箱頂
        box = px.iloc[-(BOX_DAYS + 1):-1].copy()
        box_high = float(box["high"].max())
        box_low = float(box["low"].min())
        if box_low <= 0:
            return None

        box_width = (box_high - box_low) / box_low
        high_touches = int((box["high"] >= box_high * (1 - BOX_TOUCH_TOLERANCE)).sum())
        low_touches = int((box["low"] <= box_low * (1 + BOX_TOUCH_TOLERANCE)).sum())

        box_valid = (
            box_width <= MAX_BOX_WIDTH
            and high_touches >= MIN_BOX_TOUCHES
            and low_touches >= MIN_BOX_TOUCHES
        )
        box_score = _box_score(box, box_width, high_touches, low_touches)
        box_quality = _box_quality(box_score)

        # 4. 箱底 / 突破
        distance_to_bottom = close / box_low - 1
        near_bottom = 0 <= distance_to_bottom <= BOTTOM_DISTANCE
        pullback_volume_ok = vol_20 > 0 and vol_5 <= vol_20 * BOTTOM_VOLUME_RATIO
        stabilizing = close >= open_price or close >= float(prev["close"])

        bottom_signal = (
            liquidity_ok
            and momentum_ok
            and long_trend_ok
            and box_valid
            and near_bottom
            and pullback_volume_ok
            and stabilizing
        )

        breakout_price_ok = close >= box_high * (1 + BREAKOUT_BUFFER)
        breakout_volume_ok = vol_ratio >= BREAKOUT_VOLUME_RATIO
        breakout_signal = (
            liquidity_ok
            and momentum_ok
            and long_trend_ok
            and box_valid
            and breakout_price_ok
            and breakout_volume_ok
            and close > ma20
        )

        # 5. 多重背離 + ATR 歷史統計
        div = analyze_divergence(px)
        divergence_score = int(div.get("score", 50))
        div_perf = div.get("performance", {})

        # 6. 動能 / 箱型 / 背離 三層總分
        signal_score = round(
            momentum_score * 0.45
            + box_score * 0.35
            + divergence_score * 0.20
        )

        signals: list[str] = []
        if liquidity_ok:
            signals.append("成交量≥7000張")
        if momentum_ok:
            signals.append("近期強勢動能")
        if chg_20d >= 0.20 or chg_60d >= 0.30:
            signals.append("高動能")
        if long_trend_ok:
            signals.append("中長期多頭")
        if strong_alignment:
            signals.append("均線多頭排列")
        if box_valid:
            signals.append(f"箱型成立({box_quality})")

        setup = "NONE"
        if breakout_signal:
            action = "BUY"
            setup = "BOX_BREAKOUT"
            signals.append("箱型放量突破")
        elif bottom_signal:
            action = "BUY"
            setup = "BOX_BOTTOM"
            signals.append("箱底量縮止穩")
        elif liquidity_ok and momentum_ok and long_trend_ok:
            action = "WATCH"
            setup = "HOT"
            signals.append("強勢股等待買點")
        else:
            action = "SKIP"

        # 7. 風險註記
        if not liquidity_ok:
            result["risk_notes"].append(
                f"成交量不足 7000 張，目前約 {volume / 1000:,.0f} 張"
            )
        if not momentum_ok:
            result["risk_notes"].append("近期動能不足")
        if not long_trend_ok:
            result["risk_notes"].append("中長期趨勢未達 MA60 > MA120 且 MA60 向上")
        if liquidity_ok and momentum_ok and long_trend_ok and not box_valid:
            result["risk_notes"].append("強勢股，但尚未形成穩定箱型")
        if action == "WATCH" and setup == "HOT":
            result["risk_notes"].append("等待箱底止穩或放量突破")
        if div.get("bearish_count", 0) >= 2:
            result["risk_notes"].append(
                f"近期有 {div['bearish_count']}/4 項空頭背離，追價需保守"
            )

        # 8. 原本量價分析保留
        vp = detect_patterns(px)

        # 9. 原專案固定風控先保留，避免 Performance 分頁統計口徑突然改變
        stop_loss = CONFIG["stop_loss"]
        target_return = CONFIG["target_return"]
        entry_price = close
        stop_price = round(entry_price * (1 - stop_loss), 2)
        target_price = round(entry_price * (1 + target_return), 2)
        rr = round(target_return / stop_loss, 2)
        position_pct = min(2.0 / (stop_loss * 100) * 100, 20.0)

        high_252 = float(px["high"].iloc[-252:].max()) if len(px) >= 252 else float(px["high"].max())
        pct_from_high = (close / high_252 - 1) * 100

        result.update({
            "action": action,
            "signal_score": signal_score,
            "components": {
                "fundamental_pass": None,
                "tech_score": signal_score,
                "tech_signals": signals,
                "backtest_winrate": div_perf.get("winrate"),
                "backtest_samples": div_perf.get("closed", 0),
                "momentum_score": momentum_score,
                "box_score": box_score,
                "box_quality": box_quality,
                "divergence_score": divergence_score,
                "volume_patterns": vp["patterns"],
                "volume_details": vp["details"],
                "volume_bonus": vp["bonus"],
                "volume_verdict": volume_verdict(vp["patterns"]),
            },
            "trend": {
                "setup": setup,
                "chg_5d": round(chg_5d * 100, 2),
                "chg_20d": round(chg_20d * 100, 2),
                "chg_60d": round(chg_60d * 100, 2),
                "vol_ratio": round(vol_ratio, 2),
                "volume_lots": round(volume / 1000, 0),
                "pct_from_high": round(pct_from_high, 1),
                "above_ma20": bool(close > ma20),
                "above_ma60": bool(close > ma60),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "ma120": round(ma120, 2),
                "box_high": round(box_high, 2),
                "box_low": round(box_low, 2),
                "box_width_pct": round(box_width * 100, 1),
                "box_high_touches": high_touches,
                "box_low_touches": low_touches,
                "distance_to_bottom_pct": round(distance_to_bottom * 100, 1),
            },
            "divergence": div,
            "entry_price": round(entry_price, 2),
            "stop_loss_price": stop_price,
            "target_price": target_price,
            "risk_reward_ratio": rr,
            "position_size_pct": round(position_pct, 1),
            "entry_rule": "動能箱型 V2：箱底量縮止穩 / 放量突破 + 多重背離確認",
        })

        return result

    except Exception as e:
        result["action"] = "ERROR"
        result["signal_score"] = 0
        result["risk_notes"].append(f"錯誤: {str(e)[:160]}")
        return result


# ============================================================
# Telegram 格式
# ============================================================
def _setup_name(setup: str) -> str:
    return {
        "BOX_BREAKOUT": "🚀 箱型突破",
        "BOX_BOTTOM": "🎯 箱底伏擊",
        "HOT": "🔥 強勢觀察",
        "NONE": "—",
    }.get(setup, setup)


def _div_token(name: str, direction: int) -> str:
    if direction == 1:
        return f"✅{name}"
    if direction == -1:
        return f"⚠️{name}(空)"
    return f"➖{name}"


def _fmt_num(value, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    if signed:
        return f"{float(value):+.{digits}f}"
    return f"{float(value):.{digits}f}"


def _format_one_stock(s: dict) -> list[str]:
    t = s.get("trend", {})
    c = s.get("components", {})
    d = s.get("divergence", {})
    perf = d.get("performance", {})
    current = d.get("current", {})

    action = s.get("action", "")
    icon = {
        "BUY": "🟢",
        "WATCH": "🟡",
        "SKIP": "⚪",
        "ERROR": "🔴",
    }.get(action, "⚪")

    setup = _setup_name(t.get("setup", "NONE"))

    lines = [
        f"{icon} *{s['stock_id']} {s['name']}* | {setup} | 綜合 {s.get('signal_score', 0)}/100"
    ]

    lines.append(
        f"🔥 Momentum {c.get('momentum_score', 0)}/100 | "
        f"📦 Box {c.get('box_score', 0)}/100（{c.get('box_quality', '—')}） | "
        f"🔀 Div {c.get('divergence_score', 50)}/100"
    )

    if t:
        lines.append(
            f"現價 {s.get('entry_price')} | 箱頂 {t.get('box_high')} | 箱底 {t.get('box_low')} | "
            f"距箱底 {t.get('distance_to_bottom_pct', 0):+.1f}%"
        )
        lines.append(
            f"20日 {t.get('chg_20d', 0):+.1f}% | 60日 {t.get('chg_60d', 0):+.1f}% | "
            f"量 {t.get('volume_lots', 0):,.0f} 張 | 量比 {t.get('vol_ratio', 0):.2f}x"
        )

    div_tokens = [
        _div_token(name, int(current.get(name, 0)))
        for name in ("RSI", "MFI", "MACD", "OBV")
    ]
    lines.append(
        "Bullish Divergence: "
        + " ".join(div_tokens)
        + f" | 多頭確認 {d.get('bullish_count', 0)}/4"
    )

    closed = int(perf.get("closed", 0) or 0)
    if closed > 0:
        lines.append(
            f"📈 背離歷史勝率 {_fmt_num(perf.get('winrate'))}% "
            f"(W{perf.get('wins', 0)}/L{perf.get('losses', 0)}, Open {perf.get('open', 0)}) | "
            f"損平 {_fmt_num(perf.get('break_even_winrate'))}%"
        )
        lines.append(
            f"CUM {_fmt_num(perf.get('cum_atr'), 2, signed=True)} ATR | "
            f"期望 {_fmt_num(perf.get('expectancy_atr'), 2, signed=True)} ATR/筆 | "
            f"{perf.get('grade', '—')}・樣本信心 {perf.get('confidence', '—')}"
        )
    else:
        lines.append("📈 背離歷史勝率：樣本不足，暫不評估")

    tech_signals = c.get("tech_signals", [])
    if tech_signals:
        lines.append("訊號: " + " / ".join(tech_signals))

    if action == "BUY":
        lines.append(
            f"參考風控：固定版停損 {s.get('stop_loss_price')} | 目標 {s.get('target_price')}"
        )

    if s.get("risk_notes"):
        lines.append("⚠️ " + " / ".join(s["risk_notes"]))

    return lines


def format_momentum_messages(
    signals: list[dict],
    market: dict | None = None,
    night_note: str | None = None,
) -> list[str]:
    buys = [s for s in signals if s.get("action") == "BUY"]
    watches = [s for s in signals if s.get("action") == "WATCH"]
    skips = [s for s in signals if s.get("action") in ("SKIP", "ERROR")]

    today = datetime.now().strftime("%Y/%m/%d")

    msg1 = [
        f"🔥 *動能箱型選股 V2* {today}",
        f"掃描 {len(signals)} 檔 | BUY {len(buys)} | WATCH {len(watches)} | SKIP {len(skips)}",
        "",
    ]

    if market and market.get("note"):
        msg1.extend(["🎯 *大盤濾鏡*", market["note"], ""])
    if night_note:
        msg1.extend(["🌙 *夜盤濾鏡*", night_note, ""])

    msg1.extend([
        "📋 *V2 核心*",
        "• Momentum：成交量 + 20/60日漲幅 + MA60/MA120趨勢",
        "• Box：箱寬 + 箱頂/箱底碰觸 + 區間穩定度",
        "• Divergence：RSI / MFI / MACD / OBV",
        "• 勝率提示：多頭背離以 TP 2.5 ATR / SL 1.5 ATR 統計",
        "• ATR 模型損益平衡勝率 = 37.5%",
    ])

    msg2: list[str] = []
    if buys:
        msg2.extend([f"🟢 *今日動能箱型訊號 ({len(buys)})*", ""])
        for s in buys[:5]:
            msg2.extend(_format_one_stock(s))
            msg2.append("")
    else:
        msg2.extend(["🟢 *今日沒有箱底 / 箱型突破訊號*", ""])

    if watches:
        msg2.extend([f"🔥 *強勢觀察 TOP {min(5, len(watches))}*", ""])
        for s in watches[:5]:
            msg2.extend(_format_one_stock(s))
            msg2.append("")

    msg3 = [
        "🧠 *勝率怎麼看*",
        "",
        "• 背離勝率 > 37.5%：在 2.5 ATR / 1.5 ATR 模型下開始具有正期望",
        "• CUM ATR > 0：歷史累積 ATR 單位為正；不是百分比報酬",
        "• 期望 ATR/筆 = CUM ATR ÷ 已完成訊號，越高越好",
        "• 樣本 < 8 筆時一律標示『樣本不足』，避免被小樣本誤導",
        "• 此勝率只評估背離訊號，不等於整套動能箱型策略的完整勝率",
        "",
        "_系統訊號僅供研究與紀錄，不代表保證獲利。_",
    ]

    return ["\n".join(msg1), "\n".join(msg2), "\n".join(msg3)]
