# -*- coding: utf-8 -*-
"""
因子库 v4.0 — Stratapro 扩展因子模块
======================================
基于 QuantDinger 技术指标架构，扩展 A 股多因子评分体系。
新增因子：RSI(14)、KDJ(9,3,3)、MACD(12,26,9)、布林带(20,2)、ATR(14)

作者：AtomCollide-智械工坊团队
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Wilder RSI — 与同花顺/东方财富终端一致。"""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        chg = closes[i] - closes[i - 1]
        gains.append(chg if chg > 0 else 0.0)
        losses.append(-chg if chg < 0 else 0.0)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def compute_kdj(highs: List[float], lows: List[float], closes: List[float],
                period: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """KDJ(9,3,3) — A 股终端惯例，K/D 初始值 50。"""
    n = len(closes)
    if n < period:
        return None, None, None
    k_prev, d_prev = 50.0, 50.0
    for i in range(period - 1, n):
        win_high = max(highs[j] for j in range(i - period + 1, i + 1))
        win_low = min(lows[j] for j in range(i - period + 1, i + 1))
        rsv = 50.0 if win_high == win_low else (closes[i] - win_low) / (win_high - win_low) * 100.0
        k_prev = (k_prev * (k_smooth - 1) + rsv) / k_smooth
        d_prev = (d_prev * (d_smooth - 1) + k_prev) / d_smooth
    j_val = 3.0 * k_prev - 2.0 * d_prev
    return round(k_prev, 2), round(d_prev, 2), round(j_val, 2)


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """MACD(12,26,9) — DIF/DEA/MACD柱。"""
    if len(closes) < slow + signal:
        return None, None, None
    def _ema(data, period):
        ema = [data[0]]
        k = 2.0 / (period + 1)
        for i in range(1, len(data)):
            ema.append(data[i] * k + ema[-1] * (1 - k))
        return ema
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    dif = [ema_fast[i] - ema_slow[i] for i in range(len(closes))]
    dea = _ema(dif, signal)
    macd_bar = [(dif[i] - dea[i]) * 2 for i in range(len(closes))]
    return round(dif[-1], 4), round(dea[-1], 4), round(macd_bar[-1], 4)


def compute_bollinger(closes: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """布林带(20,2) — 上轨/中轨/下轨/当前%B位置。"""
    if len(closes) < period:
        return None, None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    std = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return round(upper, 2), round(mid, 2), round(lower, 2), round(pct_b, 4)


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """ATR(14) — 平均真实波幅，衡量波动率。"""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return round(atr, 4)


def compute_momentum_score(closes: List[float], highs: List[float] = None, lows: List[float] = None) -> Dict[str, float]:
    """
    综合动量评分 — 将所有新因子合成一个 0~100 分值。
    返回各子因子得分及综合动量分。
    """
    if not closes or len(closes) < 30:
        return {'momentum': 50.0}

    # RSI 得分：30~70 映射到 0~100，超买超卖惩罚
    rsi = compute_rsi(closes)
    if rsi is not None:
        if rsi < 30:
            rsi_score = max(0, rsi / 30 * 60)  # 超卖但可能反弹
        elif rsi > 70:
            rsi_score = max(0, 100 - (rsi - 70) / 30 * 60)  # 超买风险
        else:
            rsi_score = (rsi - 30) / 40 * 100
    else:
        rsi_score = 50.0

    # MACD 得分：DIF>DEA 且 MACD柱放大 → 高分
    dif, dea, bar = compute_macd(closes) if len(closes) >= 35 else (None, None, None)
    if dif is not None:
        macd_score = 50.0
        if dif > dea:
            macd_score += 25
        else:
            macd_score -= 25
        if bar is not None and bar > 0:
            macd_score += min(25, bar * 100)
        elif bar is not None:
            macd_score += max(-25, bar * 100)
        macd_score = max(0, min(100, macd_score))
    else:
        macd_score = 50.0

    # 布林带%B 得分
    if highs and lows and len(closes) >= 20:
        _, _, _, pct_b = compute_bollinger(closes)
        if pct_b is not None:
            boll_score = max(0, min(100, pct_b * 100))
        else:
            boll_score = 50.0
    else:
        boll_score = 50.0

    # KDJ 得分
    if highs and lows and len(closes) >= 9:
        k, d, j = compute_kdj(highs, lows, closes)
        if k is not None:
            kdj_score = max(0, min(100, k))
        else:
            kdj_score = 50.0
    else:
        kdj_score = 50.0

    # ATR 波动率归一化
    if highs and lows and len(closes) >= 15:
        atr = compute_atr(highs, lows, closes)
        if atr and closes[-1] > 0:
            atr_pct = atr / closes[-1] * 100
            # 低波动(1%)=80分, 高波动(5%)=20分
            atr_score = max(0, min(100, 80 - (atr_pct - 1) * 15))
        else:
            atr_score = 50.0
    else:
        atr_score = 50.0

    momentum = rsi_score * 0.30 + macd_score * 0.25 + boll_score * 0.15 + kdj_score * 0.20 + atr_score * 0.10
    return {
        'momentum': round(momentum, 2),
        'rsi': rsi if rsi is not None else 50.0,
        'rsi_score': round(rsi_score, 2),
        'macd_dif': dif if dif is not None else 0.0,
        'macd_score': round(macd_score, 2),
        'boll_pctb': pct_b if (highs and lows and len(closes) >= 20 and (pct_b := compute_bollinger(closes)[3]) is not None) else 0.5,
        'boll_score': round(boll_score, 2),
        'kdj_k': k if (highs and lows and len(closes) >= 9 and (k := compute_kdj(highs, lows, closes)[0]) is not None) else 50.0,
        'kdj_score': round(kdj_score, 2),
        'atr_score': round(atr_score, 2),
    }
