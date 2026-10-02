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
