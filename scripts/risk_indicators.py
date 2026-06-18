# -*- coding: utf-8 -*-
"""
风险指标模块 v4.0 — Stratapro 风控增强
========================================
基于 QuantDinger risk_guard 架构，为 A 股选股系统提供：
  - 最大回撤 (Max Drawdown)
  - 年化波动率 (Annualized Volatility)
  - 夏普比率 (Sharpe Ratio)
  - 索提诺比率 (Sortino Ratio)
  - 止损跟踪 (Trailing Stop)
  - 综合风险评级 (0-100，越低越安全)

作者：AtomCollide-智械工坊团队
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math


def max_drawdown(equity_curve: List[float]) -> Dict[str, float]:
    """
    最大回撤计算。
    返回：max_dd (百分比), peak_value, trough_value, peak_idx, trough_idx
    """
    if not equity_curve or len(equity_curve) < 2:
        return {'max_dd': 0.0, 'peak_value': 0, 'trough_value': 0, 'peak_idx': 0, 'trough_idx': 0}
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    dd_peak_idx, dd_trough_idx = 0, 0
    for i, val in enumerate(equity_curve):
        if val > peak:
            peak = val
            peak_idx = i
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            dd_peak_idx = peak_idx
            dd_trough_idx = i
    return {
        'max_dd': round(max_dd * 100, 2),
        'peak_value': round(equity_curve[dd_peak_idx], 2),
        'trough_value': round(equity_curve[dd_trough_idx], 2),
        'peak_idx': dd_peak_idx,
        'trough_idx': dd_trough_idx,
    }


def annualized_volatility(returns: List[float], trading_days: int = 245) -> float:
    """年化波动率。returns 为日收益率列表（小数形式）。"""
    if not returns or len(returns) < 5:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    daily_vol = math.sqrt(var)
    ann_vol = daily_vol * math.sqrt(trading_days)
    return round(ann_vol * 100, 2)


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.025, trading_days: int = 245) -> float:
    """
    夏普比率。risk_free_rate 为年化无风险收益率（默认 2.5%，对应中国国债）。
    """
    if not returns or len(returns) < 10:
        return 0.0
    mean_annual = sum(returns) / len(returns) * trading_days
    var = sum((r - sum(returns)/len(returns)) ** 2 for r in returns) / (len(returns) - 1)
    daily_vol = math.sqrt(var)
    ann_vol = daily_vol * math.sqrt(trading_days)
    if ann_vol == 0:
        return 0.0
    return round((mean_annual - risk_free_rate) / ann_vol, 4)


def sortino_ratio(returns: List[float], risk_free_rate: float = 0.025, trading_days: int = 245) -> float:
    """索提诺比率 — 只惩罚下行波动。"""
    if not returns or len(returns) < 10:
        return 0.0
    mean_annual = sum(returns) / len(returns) * trading_days
    downside = [r for r in returns if r < 0]
    if not downside or len(downside) < 3:
        return 10.0  # 无下行风险
    down_var = sum(r ** 2 for r in downside) / len(downside)
    down_vol = math.sqrt(down_var) * math.sqrt(trading_days)
    if down_vol == 0:
        return 10.0
    return round((mean_annual - risk_free_rate) / down_vol, 4)


def trailing_stop_check(current_price: float, entry_price: float,
                        stop_pct: float = 8.0, take_profit_pct: float = 20.0) -> Dict[str, object]:
    """
    止损/止盈跟踪检查。
    返回：是否触发止损/止盈、当前盈亏%、建议操作。
    """
    if entry_price <= 0:
        return {'triggered': False, 'pnl_pct': 0.0, 'action': '持有', 'reason': ''}
    pnl_pct = (current_price - entry_price) / entry_price * 100
    result = {
        'triggered': False,
        'pnl_pct': round(pnl_pct, 2),
        'action': '持有',
        'reason': '',
        'entry_price': round(entry_price, 2),
        'current_price': round(current_price, 2),
    }
    if pnl_pct <= -stop_pct:
        result['triggered'] = True
        result['action'] = '止损出局'
        result['reason'] = f'亏损{abs(pnl_pct):.1f}%已达止损线{stop_pct}%'
    elif pnl_pct >= take_profit_pct:
        result['action'] = '考虑止盈'
        result['reason'] = f'盈利{pnl_pct:.1f}%已达止盈线{take_profit_pct}%'
    elif pnl_pct <= -stop_pct * 0.6:
        result['action'] = '关注风险'
        result['reason'] = f'亏损{abs(pnl_pct):.1f}%接近止损线'
    return result


def calculate_risk_score(closes: List[float], entry_price: float = None) -> Dict[str, object]:
    """
    综合风险评级（0-100，越低越安全）。
    维度：波动率(40%) + 回撤(30%) + 趋势(30%)
    """
    if not closes or len(closes) < 20:
        return {'risk_score': 50, 'risk_level': '中', 'detail': '数据不足'}

    # 计算日收益率
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
    if not returns:
        return {'risk_score': 50, 'risk_level': '中', 'detail': '收益率计算失败'}

    # 1. 波动率评分 (0-100, 低波动=低分=安全)
    ann_vol = annualized_volatility(returns)
    vol_score = min(100, max(0, ann_vol * 1.5))  # 20%波动 → 30分, 40%→60分

    # 2. 回撤评分
    # 用最近 60 日价格估算回撤
    recent = closes[-min(60, len(closes)):]
    dd_info = max_drawdown(recent)
    dd_score = min(100, max(0, dd_info['max_dd'] * 3))  # 10%回撤→30分, 20%→60分

    # 3. 趋势评分（价格在近期高低点中的位置）
    high_60 = max(recent)
    low_60 = min(recent)
    if high_60 > low_60:
        position = (closes[-1] - low_60) / (high_60 - low_60)
        # 位置越低越危险
        trend_score = max(0, min(100, (1 - position) * 100))
    else:
        trend_score = 50

    risk_score = vol_score * 0.4 + dd_score * 0.3 + trend_score * 0.3
    risk_score = round(risk_score, 1)

    if risk_score < 25:
        risk_level = '低风险'
    elif risk_score < 45:
        risk_level = '较低风险'
    elif risk_score < 65:
        risk_level = '中等风险'
    elif risk_score < 80:
        risk_level = '较高风险'
    else:
        risk_level = '高风险'

    result = {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'ann_volatility': ann_vol,
        'max_drawdown_60d': dd_info['max_dd'],
        'price_position_60d': round(position * 100, 1) if high_60 > low_60 else 50.0,
        'vol_score': round(vol_score, 1),
        'dd_score': round(dd_score, 1),
        'trend_score': round(trend_score, 1),
    }

    # 止损检查（如果有入场价）
    if entry_price and entry_price > 0:
        stop_info = trailing_stop_check(closes[-1], entry_price)
        result['stop_check'] = stop_info

    return result


def format_risk_report(closes: List[float], stock_name: str = '', entry_price: float = None) -> str:
    """格式化风险报告文本。"""
    risk = calculate_risk_score(closes, entry_price)
    lines = [f'=== 风险评估 {stock_name} ===']
    lines.append(f'综合风险: {risk["risk_score"]}/100 ({risk["risk_level"]})')
    if 'ann_volatility' in risk:
        lines.append(f'年化波动率: {risk["ann_volatility"]}%')
        lines.append(f'60日最大回撤: {risk["max_drawdown_60d"]}%')
        lines.append(f'60日价格位置: {risk["price_position_60d"]}%')
    if 'stop_check' in risk:
        sc = risk['stop_check']
        lines.append(f'持仓盈亏: {sc["pnl_pct"]:+.2f}%  建议: {sc["action"]}')
        if sc['reason']:
            lines.append(f'  原因: {sc["reason"]}')
    return '\n'.join(lines)
