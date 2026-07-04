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


# ── Qlib-Inspired Multi-Period Alpha Factors ─────────────────────────────
# 以下因子基于 Qlib Alpha158 因子集设计，覆盖动量、波动率、量价、
# 趋势和均值回复五类维度，用于增强 Stratapro 多因子评分体系。

def compute_roc_multi(closes: List[float], periods: List[int] = None) -> Optional[Dict[str, float]]:
    """多周期变化率 (Rate of Change) — 动量因子。
    
    ROC = (close[t] - close[t-n]) / close[t-n] * 100
    
    Args:
        closes: 收盘价序列
        periods: 计算周期列表，默认 [5, 10, 20]
    
    Returns:
        dict 各周期 ROC 值，或 None 数据不足时
    """
    if periods is None:
        periods = [5, 10, 20]
    if not closes or len(closes) < max(periods) + 1:
        return None
    result = {}
    for p in periods:
        if len(closes) > p and closes[-p - 1] != 0:
            roc = (closes[-1] - closes[-p - 1]) / closes[-p - 1] * 100.0
            result[f'roc_{p}'] = round(roc, 4)
        else:
            result[f'roc_{p}'] = 0.0
    return result


def compute_max_drawdown(closes: List[float], period: int = 20) -> Optional[float]:
    """最大回撤 (Maximum Drawdown) — 风险/波动率因子。
    
    在过去 period 个交易日内，从峰值到谷底的最大回撤百分比。
    
    Args:
        closes: 收盘价序列
        period: 回看窗口，默认 20
    
    Returns:
        最大回撤百分比(正数表示回撤幅度)，或 None
    """
    if not closes or len(closes) < period:
        return None
    window = closes[-period:]
    peak = window[0]
    max_dd = 0.0
    for price in window:
        if price > peak:
            peak = price
        dd = (peak - price) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100.0, 4)


def compute_volume_price_corr(closes: List[float], volumes: List[float], period: int = 20) -> Optional[float]:
    """量价相关系数 — 量价因子。
    
    计算最近 period 天收盘价变动与成交量的皮尔逊相关系数。
    正相关表示价涨量增（趋势确认），负相关表示背离。
    
    Args:
        closes: 收盘价序列
        volumes: 成交量序列
        period: 回看窗口，默认 20
    
    Returns:
        相关系数 [-1, 1]，或 None
    """
    if not closes or not volumes or len(closes) < period + 1 or len(volumes) < period + 1:
        return None
    price_changes = np.array([closes[i] - closes[i - 1] for i in range(-period, 0)])
    vol_window = np.array(volumes[-period:])
    # 标准差为零时相关系数无意义
    if np.std(price_changes) < 1e-10 or np.std(vol_window) < 1e-10:
        return 0.0
    corr = np.corrcoef(price_changes, vol_window)[0, 1]
    return round(float(corr), 4) if not np.isnan(corr) else 0.0


def compute_volatility_ratio(closes: List[float], short: int = 5, long: int = 20) -> Optional[float]:
    """波动率比率 (Volatility Ratio) — 波动率因子。
    
    短期波动率 / 长期波动率。>1 表示近期波动放大，<1 表示收敛。
    
    Args:
        closes: 收盘价序列
        short: 短期窗口，默认 5
        long: 长期窗口，默认 20
    
    Returns:
        波动率比率，或 None
    """
    if not closes or len(closes) < long + 1:
        return None
    returns = np.array([(closes[i] - closes[i - 1]) / closes[i - 1]
                        for i in range(-long, 0) if closes[i - 1] != 0])
    if len(returns) < long:
        return None
    short_vol = float(np.std(returns[-short:]))
    long_vol = float(np.std(returns))
    if long_vol < 1e-10:
        return 1.0
    return round(short_vol / long_vol, 4)


def compute_williams_r(highs: List[float], lows: List[float], closes: List[float],
                       period: int = 14) -> Optional[float]:
    """威廉指标 (Williams %R) — 趋势/超买超卖因子。
    
    WR = (HH - Close) / (HH - LL) * (-100)
    范围 [-100, 0]，-20 以上超买，-80 以下超卖。
    
    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        period: 回看窗口，默认 14
    
    Returns:
        Williams %R 值 [-100, 0]，或 None
    """
    if not closes or not highs or not lows:
        return None
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return None
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    if hh == ll:
        return -50.0
    wr = (hh - closes[-1]) / (hh - ll) * -100.0
    return round(wr, 2)


def compute_cci(highs: List[float], lows: List[float], closes: List[float],
                period: int = 20) -> Optional[float]:
    """商品通道指数 (Commodity Channel Index) — 趋势因子。
    
    CCI = (TP - SMA(TP)) / (0.015 * MeanDeviation)
    TP = (High + Low + Close) / 3
    
    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        period: 回看窗口，默认 20
    
    Returns:
        CCI 值，或 None
    """
    if not closes or not highs or not lows:
        return None
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return None
    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(-period, 0)]
    sma_tp = sum(tp) / period
    mean_dev = sum(abs(t - sma_tp) for t in tp) / period
    if mean_dev < 1e-10:
        return 0.0
    cci = (tp[-1] - sma_tp) / (0.015 * mean_dev)
    return round(cci, 2)


def compute_obv_score(closes: List[float], volumes: List[float]) -> Optional[float]:
    """能量潮趋势评分 (On-Balance Volume Score) — 量价因子。
    
    计算 OBV 的线性回归斜率，归一化到 0~100。
    OBV 上升表示资金流入，下降表示流出。
    
    Args:
        closes: 收盘价序列
        volumes: 成交量序列
    
    Returns:
        OBV 趋势评分 [0, 100]，或 None
    """
    if not closes or not volumes or len(closes) < 10 or len(volumes) < 10:
        return None
    n = min(len(closes), len(volumes))
    obv = [0.0]
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    # 线性回归斜率
    obv_arr = np.array(obv[-20:]) if len(obv) >= 20 else np.array(obv)
    x = np.arange(len(obv_arr), dtype=float)
    if np.std(obv_arr) < 1e-10:
        return 50.0
    slope = float(np.polyfit(x, obv_arr, 1)[0])
    # 归一化到 0~100：用 sigmoid 映射
    norm_slope = slope / (np.std(obv_arr) + 1e-10)
    score = 100.0 / (1.0 + np.exp(-norm_slope))
    return round(float(score), 2)


def enhanced_momentum_score(closes: List[float], highs: List[float] = None,
                            lows: List[float] = None,
                            volumes: List[float] = None) -> Dict[str, float]:
    """
    增强动量评分 (Enhanced Momentum Score) — Qlib Alpha158 风格。
    
    在原有 momentum_score 基础上新增五类因子维度：
    - 动量：多周期 ROC
    - 波动率：最大回撤、波动率比率
    - 量价：量价相关、OBV 趋势
    - 趋势：Williams %R、CCI
    - 均值回复：布林带 %B (复用已有)
    
    返回所有子因子得分及综合评分 (0~100)。
    """
    # 基础评分
    base = compute_momentum_score(closes, highs, lows)
    
    # 新增因子默认值
    roc_score = 50.0
    drawdown_score = 50.0
    vol_price_score = 50.0
    vol_ratio_score = 50.0
    williams_score = 50.0
    cci_score = 50.0
    obv_trend_score = 50.0
    
    # ── 动量：多周期 ROC ──
    roc = compute_roc_multi(closes) if closes and len(closes) >= 21 else None
    if roc:
        # 综合多周期 ROC：短周期权重高
        roc_vals = list(roc.values())
        roc_avg = sum(roc_vals) / len(roc_vals) if roc_vals else 0.0
        # 映射：ROC -10%~+10% → 0~100
        roc_score = max(0, min(100, (roc_avg + 10) / 20 * 100))
    
    # ── 波动率：最大回撤 ──
    mdd = compute_max_drawdown(closes) if closes and len(closes) >= 20 else None
    if mdd is not None:
        # 回撤越小越好：0%→100分, 20%→0分
        drawdown_score = max(0, min(100, 100 - mdd * 5))
    
    # ── 波动率：波动率比率 ──
    vol_ratio = compute_volatility_ratio(closes) if closes and len(closes) >= 21 else None
    if vol_ratio is not None:
        # 比率 0.5~2.0 → 100~0（波动放大是风险信号）
        vol_ratio_score = max(0, min(100, (2.0 - vol_ratio) / 1.5 * 100))
    
    # ── 量价相关 ──
    if closes and volumes and len(closes) >= 21 and len(volumes) >= 21:
        vpc = compute_volume_price_corr(closes, volumes)
        if vpc is not None:
            # 正相关→高分，负相关→低分：-1~+1 → 0~100
            vol_price_score = max(0, min(100, (vpc + 1) / 2 * 100))
    
    # ── 趋势：Williams %R ──
    if highs and lows and closes and len(closes) >= 14:
        wr = compute_williams_r(highs, lows, closes)
        if wr is not None:
            # WR 范围 [-100, 0]，映射到 [0, 100]
            williams_score = max(0, min(100, wr + 100))
    
    # ── 趋势：CCI ──
    if highs and lows and closes and len(closes) >= 20:
        cci = compute_cci(highs, lows, closes)
        if cci is not None:
            # CCI 范围通常 [-300, 300]，映射到 [0, 100]
            cci_score = max(0, min(100, (cci + 300) / 600 * 100))
    
    # ── 量价：OBV 趋势 ──
    if closes and volumes:
        obv_s = compute_obv_score(closes, volumes)
        if obv_s is not None:
            obv_trend_score = obv_s
    
    # ── 综合评分 ──
    # 权重分配：动量30%, 波动率20%, 量价15%, 趋势20%, 均值回复15%
    enhanced = (
        roc_score * 0.15 +               # 动量子项1
        base['momentum'] * 0.15 +         # 动量子项2（原综合）
        (drawdown_score * 0.10 +
         vol_ratio_score * 0.10) +        # 波动率
        (vol_price_score * 0.075 +
         obv_trend_score * 0.075) +       # 量价
        (williams_score * 0.10 +
         cci_score * 0.10) +              # 趋势
        base.get('boll_score', 50.0) * 0.15  # 均值回复
    )
    
    result = dict(base)
    result.update({
        'roc': roc if roc else {},
        'roc_score': round(roc_score, 2),
        'max_drawdown': mdd if mdd is not None else 0.0,
        'drawdown_score': round(drawdown_score, 2),
        'vol_price_corr': vol_price_score,
        'vol_price_score': round(vol_price_score, 2),
        'volatility_ratio': vol_ratio if vol_ratio is not None else 1.0,
        'vol_ratio_score': round(vol_ratio_score, 2),
        'williams_r': williams_score,
        'williams_score': round(williams_score, 2),
        'cci': cci_score,
        'cci_score': round(cci_score, 2),
        'obv_score': round(obv_trend_score, 2),
        'enhanced_momentum': round(enhanced, 2),
    })
    return result
