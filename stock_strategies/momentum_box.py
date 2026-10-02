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
