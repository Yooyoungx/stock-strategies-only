from __future__ import annotations

import math
from typing import Any

import pandas as pd

PIVOT_LEN = 5
ALIGN_WINDOW = 3
RECENT_BARS = 15
ATR_PERIOD = 14
TP_ATR_MULT = 2.5
SL_ATR_MULT = 1.5
MIN_CLOSED_SAMPLES = 8

INDICATOR_WEIGHTS = {
    "RSI": 20,
    "MFI": 20,
    "MACD": 20,
    "OBV": 40,
}


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _atr(df: pd.DataFrame, length: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _rma(tr, length)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, math.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    return rsi.where(~both_zero, 50.0)


def _mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * df["volume"]
    change = typical.diff()
    pos_flow = raw_flow.where(change > 0, 0.0)
    neg_flow = raw_flow.where(change < 0, 0.0)
    pos_sum = pos_flow.rolling(length, min_periods=length).sum()
    neg_sum = neg_flow.rolling(length, min_periods=length).sum()
    ratio = pos_sum / neg_sum.replace(0, math.nan)
    mfi = 100 - (100 / (1 + ratio))
    mfi = mfi.where(neg_sum != 0, 100.0)
    both_zero = (pos_sum == 0) & (neg_sum == 0)
    return mfi.where(~both_zero, 50.0)


def _macd(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    return ema12 - ema26


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = df["close"].diff().apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    )
    return (direction.fillna(0.0) * df["volume"]).cumsum()


def prepare_divergence_frame(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy().reset_index(drop=True)
    df["atr14"] = _atr(df)
    df["rsi14"] = _rsi(df["close"])
    df["mfi14"] = _mfi(df)
    df["macd"] = _macd(df["close"])
    df["obv"] = _obv(df)
    return df


def _compress_pivots(df: pd.DataFrame, candidates: list[int], kind: str, piv_len: int) -> list[int]:
    result: list[int] = []
    price_col = "low" if kind == "low" else "high"

    for idx in candidates:
        if not result or idx - result[-1] > piv_len:
            result.append(idx)
            continue

        prev_idx = result[-1]
        cur_price = float(df.iloc[idx][price_col])
        prev_price = float(df.iloc[prev_idx][price_col])

        should_replace = cur_price <= prev_price if kind == "low" else cur_price >= prev_price
        if should_replace:
            result[-1] = idx

    return result


def _price_pivots(df: pd.DataFrame, kind: str, piv_len: int) -> list[int]:
    price_col = "low" if kind == "low" else "high"
    series = df[price_col]
    width = piv_len * 2 + 1

    if kind == "low":
        rolling = series.rolling(width, center=True).min()
        mask = series.eq(rolling)
    else:
        rolling = series.rolling(width, center=True).max()
        mask = series.eq(rolling)

    candidates = [int(i) for i in df.index[mask.fillna(False)].tolist()]
    return _compress_pivots(df, candidates, kind, piv_len)


def _aligned_osc_value(series: pd.Series, pivot_idx: int, kind: str, align_window: int) -> float | None:
    left = max(0, pivot_idx - align_window)
    right = min(len(series), pivot_idx + align_window + 1)
    window = pd.to_numeric(series.iloc[left:right], errors="coerce").dropna()
    if window.empty:
        return None
    return float(window.min() if kind == "low" else window.max())


def divergence_events(
    df: pd.DataFrame,
    osc_col: str,
    piv_len: int = PIVOT_LEN,
    align_window: int = ALIGN_WINDOW,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    lows = _price_pivots(df, "low", piv_len)
    highs = _price_pivots(df, "high", piv_len)

    for prev_idx, cur_idx in zip(lows, lows[1:]):
        signal_bar = cur_idx + piv_len
        if signal_bar >= len(df):
            continue
        prev_price = float(df.iloc[prev_idx]["low"])
        cur_price = float(df.iloc[cur_idx]["low"])
        prev_osc = _aligned_osc_value(df[osc_col], prev_idx, "low", align_window)
        cur_osc = _aligned_osc_value(df[osc_col], cur_idx, "low", align_window)
        if prev_osc is None or cur_osc is None:
            continue
        if cur_price < prev_price and cur_osc > prev_osc:
            events.append(
                {
                    "direction": 1,
                    "pivot_bar": cur_idx,
                    "signal_bar": signal_bar,
                    "price_prev": prev_price,
                    "price_cur": cur_price,
                    "osc_prev": prev_osc,
                    "osc_cur": cur_osc,
                }
            )

    for prev_idx, cur_idx in zip(highs, highs[1:]):
        signal_bar = cur_idx + piv_len
        if signal_bar >= len(df):
            continue
        prev_price = float(df.iloc[prev_idx]["high"])
        cur_price = float(df.iloc[cur_idx]["high"])
        prev_osc = _aligned_osc_value(df[osc_col], prev_idx, "high", align_window)
        cur_osc = _aligned_osc_value(df[osc_col], cur_idx, "high", align_window)
        if prev_osc is None or cur_osc is None:
            continue
        if cur_price > prev_price and cur_osc < prev_osc:
            events.append(
                {
                    "direction": -1,
                    "pivot_bar": cur_idx,
                    "signal_bar": signal_bar,
                    "price_prev": prev_price,
                    "price_cur": cur_price,
                    "osc_prev": prev_osc,
                    "osc_cur": cur_osc,
                }
            )

    return sorted(events, key=lambda e: int(e["signal_bar"]))


def _resolve_long_event(
    df: pd.DataFrame,
    signal_bar: int,
    tp_mult: float,
    sl_mult: float,
) -> str:
    if signal_bar < 0 or signal_bar >= len(df):
        return "OPEN"

    entry = float(df.iloc[signal_bar]["close"])
    atr = float(df.iloc[signal_bar]["atr14"])
    if not math.isfinite(entry) or not math.isfinite(atr) or atr <= 0:
        return "OPEN"

    tp = entry + atr * tp_mult
    sl = entry - atr * sl_mult

    for j in range(signal_bar + 1, len(df)):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])
        hit_tp = high >= tp
        hit_sl = low <= sl

        if hit_tp and hit_sl:
            # 日 K 無法知道先後，保守視為停損，避免績效偏樂觀。
            return "LOSS"
        if hit_sl:
            return "LOSS"
        if hit_tp:
            return "WIN"

    return "OPEN"


def _sample_confidence(closed: int) -> str:
    if closed < MIN_CLOSED_SAMPLES:
        return "樣本不足"
    if closed < 20:
        return "低"
    if closed < 50:
        return "中"
    return "高"


def _performance_grade(winrate: float | None, expectancy: float | None, closed: int, break_even: float) -> str:
    if winrate is None or expectancy is None or closed < MIN_CLOSED_SAMPLES:
        return "⚪ 樣本不足"
    if winrate < break_even or expectancy < 0:
        return "🔴 低於損平線"
    if expectancy < 0.30:
        return "🟡 微幅正期望"
    if expectancy < 0.70:
        return "🟢 正期望"
    return "🟢 強勢正期望"


def analyze_divergence(
    px: pd.DataFrame,
    piv_len: int = PIVOT_LEN,
    recent_bars: int = RECENT_BARS,
    tp_atr_mult: float = TP_ATR_MULT,
    sl_atr_mult: float = SL_ATR_MULT,
) -> dict[str, Any]:
    df = prepare_divergence_frame(px)

    osc_map = {
        "RSI": "rsi14",
        "MFI": "mfi14",
        "MACD": "macd",
        "OBV": "obv",
    }

    all_events: dict[str, list[dict[str, Any]]] = {}
    current: dict[str, int] = {}
    per_indicator: dict[str, dict[str, Any]] = {}

    latest_bar = len(df) - 1
    cutoff = max(0, latest_bar - recent_bars)

    total_wins = 0
    total_losses = 0
    total_open = 0

    for name, col in osc_map.items():
        events = divergence_events(df, col, piv_len=piv_len)
        all_events[name] = events

        recent = [e for e in events if int(e["signal_bar"]) >= cutoff]
        current[name] = int(recent[-1]["direction"]) if recent else 0

        wins = 0
        losses = 0
        opened = 0

        for event in events:
            if int(event["direction"]) != 1:
                continue
            outcome = _resolve_long_event(
                df,
                int(event["signal_bar"]),
                tp_atr_mult,
                sl_atr_mult,
            )
            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
            else:
                opened += 1

        closed = wins + losses
        winrate = round(wins / closed * 100.0, 1) if closed else None
        cum_atr = wins * tp_atr_mult - losses * sl_atr_mult
        expectancy = cum_atr / closed if closed else None

        per_indicator[name] = {
            "wins": wins,
            "losses": losses,
            "open": opened,
            "closed": closed,
            "winrate": winrate,
            "cum_atr": round(cum_atr, 2),
            "expectancy_atr": round(expectancy, 2) if expectancy is not None else None,
        }

        total_wins += wins
        total_losses += losses
        total_open += opened

    bull_names = [name for name, direction in current.items() if direction == 1]
    bear_names = [name for name, direction in current.items() if direction == -1]
    bull_weight = sum(INDICATOR_WEIGHTS[name] for name in bull_names)
    bear_weight = sum(INDICATOR_WEIGHTS[name] for name in bear_names)

    # 50 = 中性；多頭背離往 100 推，空頭背離往 0 拉。
    divergence_score = max(0, min(100, round(50 + (bull_weight - bear_weight) / 2)))

    closed = total_wins + total_losses
    winrate = round(total_wins / closed * 100.0, 1) if closed else None
    cum_atr = total_wins * tp_atr_mult - total_losses * sl_atr_mult
    expectancy = cum_atr / closed if closed else None
    break_even = sl_atr_mult / (tp_atr_mult + sl_atr_mult) * 100.0

    performance = {
        "wins": total_wins,
        "losses": total_losses,
        "open": total_open,
        "closed": closed,
        "winrate": winrate,
        "cum_atr": round(cum_atr, 2),
        "expectancy_atr": round(expectancy, 2) if expectancy is not None else None,
        "break_even_winrate": round(break_even, 1),
        "grade": _performance_grade(winrate, expectancy, closed, break_even),
        "confidence": _sample_confidence(closed),
        "tp_atr_mult": tp_atr_mult,
        "sl_atr_mult": sl_atr_mult,
    }

    return {
        "current": current,
        "bullish": bull_names,
        "bearish": bear_names,
        "bullish_count": len(bull_names),
        "bearish_count": len(bear_names),
        "bullish_weight": bull_weight,
        "bearish_weight": bear_weight,
        "score": divergence_score,
        "performance": performance,
        "per_indicator": per_indicator,
    }
