# -*- coding: utf-8 -*-
"""
ML Ensemble Scorer — Stratapro 轻量级多因子机器学习评分器
=========================================================
基于 Qlib Alpha158 因子集的逻辑回归集成模型。
使用硬编码系数（无需 sklearn），纯 numpy 实现。

因子权重来源：Qlib Alpha158 特征重要性分析
- 动量 (momentum):     30%
- 波动率 (volatility): 20%
- 量价 (volume-price): 15%
- 趋势 (trend):       20%
- 均值回复 (mean-reversion): 15%

作者：AtomCollide-智械工坊团队
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

# 导入因子库
from factor_library import (
    compute_rsi,
    compute_kdj,
    compute_macd,
    compute_bollinger,
    compute_atr,
    compute_momentum_score,
    compute_roc_multi,
    compute_max_drawdown,
    compute_volume_price_corr,
    compute_volatility_ratio,
    compute_williams_r,
    compute_cci,
    compute_obv_score,
    enhanced_momentum_score,
)


# ── Qlib Alpha158 风格特征权重 ──────────────────────────────────────────
# 权重基于特征重要性排序，归一化到各维度
FEATURE_WEIGHTS = {
    # 动量维度 (30%)
    'rsi_score':           0.08,
    'macd_score':          0.07,
    'roc_score':           0.08,
    'kdj_score':           0.07,
    # 波动率维度 (20%)
    'atr_score':           0.07,
    'drawdown_score':      0.07,
    'vol_ratio_score':     0.06,
    # 量价维度 (15%)
    'vol_price_score':     0.08,
    'obv_score':           0.07,
    # 趋势维度 (20%)
    'williams_score':      0.07,
    'cci_score':           0.07,
    'boll_score':          0.06,
    # 均值回复维度 (15%) — boll 已计入趋势，此处用 momentum 兜底
    'momentum':            0.15,
}

# 逻辑回归截距 (bias term) — 设为 0 使中性特征得分 ≈ 50
BIAS = 0.0

# sigmoid 温度系数 — 校准：avg_feature=70→~77, avg_feature=50→50, avg_feature=30→~23
TEMPERATURE = 3.0


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid 函数。"""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


def build_feature_vector(closes: List[float], highs: List[float] = None,
                         lows: List[float] = None,
                         volumes: List[float] = None) -> Dict[str, float]:
    """
    构建完整特征向量 — 计算所有因子评分并返回扁平字典。
    
    Args:
        closes: 收盘价序列
        highs: 最高价序列（可选）
        lows: 最低价序列（可选）
        volumes: 成交量序列（可选）
    
    Returns:
        dict: 特征名 → 评分值 (0~100)
    """
    if not closes or len(closes) < 5:
        return {k: 50.0 for k in FEATURE_WEIGHTS}
    
    features = {}
    
    # ── 原始因子 ──
    # RSI
    rsi = compute_rsi(closes)
    if rsi is not None:
        features['rsi_score'] = max(0, min(100, rsi))
    else:
        features['rsi_score'] = 50.0
    
    # MACD
    if len(closes) >= 35:
        dif, dea, bar = compute_macd(closes)
        if dif is not None:
            macd_s = 50.0
            macd_s += 25 if dif > dea else -25
            if bar is not None:
                macd_s += max(-25, min(25, bar * 100))
            features['macd_score'] = max(0, min(100, macd_s))
        else:
            features['macd_score'] = 50.0
    else:
        features['macd_score'] = 50.0
    
    # KDJ
    if highs and lows and len(closes) >= 9:
        k, d, j = compute_kdj(highs, lows, closes)
        features['kdj_score'] = max(0, min(100, k)) if k is not None else 50.0
    else:
        features['kdj_score'] = 50.0
    
    # Bollinger %B
    if len(closes) >= 20:
        _, _, _, pct_b = compute_bollinger(closes)
        features['boll_score'] = max(0, min(100, pct_b * 100)) if pct_b is not None else 50.0
    else:
        features['boll_score'] = 50.0
    
    # ATR
    if highs and lows and len(closes) >= 15:
        atr = compute_atr(highs, lows, closes)
        if atr and closes[-1] > 0:
            atr_pct = atr / closes[-1] * 100
            features['atr_score'] = max(0, min(100, 80 - (atr_pct - 1) * 15))
        else:
            features['atr_score'] = 50.0
    else:
        features['atr_score'] = 50.0
    
    # ── Qlib 新增因子 ──
    # ROC 多周期
    if len(closes) >= 21:
        roc = compute_roc_multi(closes)
        if roc:
            roc_avg = sum(roc.values()) / len(roc.values())
            features['roc_score'] = max(0, min(100, (roc_avg + 10) / 20 * 100))
        else:
            features['roc_score'] = 50.0
    else:
        features['roc_score'] = 50.0
    
    # 最大回撤
    if len(closes) >= 20:
        mdd = compute_max_drawdown(closes)
        features['drawdown_score'] = max(0, min(100, 100 - mdd * 5)) if mdd is not None else 50.0
    else:
        features['drawdown_score'] = 50.0
    
    # 波动率比率
    if len(closes) >= 21:
        vr = compute_volatility_ratio(closes)
        features['vol_ratio_score'] = max(0, min(100, (2.0 - vr) / 1.5 * 100)) if vr is not None else 50.0
    else:
        features['vol_ratio_score'] = 50.0
    
    # 量价相关
    if closes and volumes and len(closes) >= 21 and len(volumes) >= 21:
        vpc = compute_volume_price_corr(closes, volumes)
        features['vol_price_score'] = max(0, min(100, (vpc + 1) / 2 * 100)) if vpc is not None else 50.0
    else:
        features['vol_price_score'] = 50.0
    
    # Williams %R
    if highs and lows and len(closes) >= 14:
        wr = compute_williams_r(highs, lows, closes)
        features['williams_score'] = max(0, min(100, wr + 100)) if wr is not None else 50.0
    else:
        features['williams_score'] = 50.0
    
    # CCI
    if highs and lows and len(closes) >= 20:
        cci = compute_cci(highs, lows, closes)
        features['cci_score'] = max(0, min(100, (cci + 300) / 600 * 100)) if cci is not None else 50.0
    else:
        features['cci_score'] = 50.0
    
    # OBV 趋势
    if closes and volumes:
        obv_s = compute_obv_score(closes, volumes)
        features['obv_score'] = obv_s if obv_s is not None else 50.0
    else:
        features['obv_score'] = 50.0
    
    # 综合动量
    if len(closes) >= 30:
        em = enhanced_momentum_score(closes, highs, lows, volumes)
        features['momentum'] = em.get('enhanced_momentum', 50.0)
    else:
        features['momentum'] = 50.0
    
    return features


def ml_ensemble_score(features: Dict[str, float]) -> float:
    """
    逻辑回归集成评分 — 使用硬编码系数计算加权评分。
    
    模型: score = sigmoid(w·x + b) * 100
    
    Args:
        features: build_feature_vector 返回的特征字典
    
    Returns:
        综合评分 [0, 100]
    """
    if not features:
        return 50.0
    
    # 加权求和
    z = BIAS
    for feat_name, weight in FEATURE_WEIGHTS.items():
        value = features.get(feat_name, 50.0)
        # 特征归一化到 [-1, 1] 范围（原始范围 [0, 100]）
        normalized = (value - 50.0) / 50.0
        z += weight * normalized
    
    # sigmoid 映射到 [0, 100]
    score = _sigmoid(z * TEMPERATURE) * 100
    return round(float(score), 2)


def score_stock(closes: List[float], highs: List[float] = None,
                lows: List[float] = None,
                volumes: List[float] = None) -> Dict[str, float]:
    """
    一站式股票评分 — 计算所有因子 + ML 集成评分。
    
    Args:
        closes: 收盘价序列
        highs: 最高价序列（可选）
        lows: 最低价序列（可选）
        volumes: 成交量序列（可选）
    
    Returns:
        dict: 包含所有子因子评分、ML 综合评分及信号建议
    """
    features = build_feature_vector(closes, highs, lows, volumes)
    ml_score = ml_ensemble_score(features)
    
    # 信号建议
    if ml_score >= 70:
        signal = 'bullish'
        confidence = 'high'
    elif ml_score >= 55:
        signal = 'bullish'
        confidence = 'medium'
    elif ml_score >= 45:
        signal = 'neutral'
        confidence = 'low'
    elif ml_score >= 30:
        signal = 'bearish'
        confidence = 'medium'
    else:
        signal = 'bearish'
        confidence = 'high'
    
    result = dict(features)
    result['ml_score'] = ml_score
    result['signal'] = signal
    result['confidence'] = confidence
    return result
