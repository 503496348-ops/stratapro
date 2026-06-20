# -*- coding: utf-8 -*-
"""
Stratapro MCP Server — AI Agent Gateway
========================================
Exposes Stratapro's quant tools as MCP (Model Context Protocol) tools.
Compatible with Cursor, Claude Code, Codex, and any MCP client.

Inspired by QuantDinger's agent-native architecture, tailored for A-stock.

Author: AtomCollide-智械工坊团队
Version: 4.0
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional import — gracefully degrade if mcp SDK not installed
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

# ---------------------------------------------------------------------------
# Internal imports (Stratapro modules)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

try:
    from factor_library import compute_rsi, compute_kdj, compute_macd, compute_bollinger, compute_atr, compute_momentum_score
except ImportError:
    compute_rsi = compute_kdj = compute_macd = compute_bollinger = compute_atr = compute_momentum_score = None

try:
    from risk_indicators import calculate_risk_score, format_risk_report, trailing_stop_check, max_drawdown, sharpe_ratio, annualized_volatility
except ImportError:
    calculate_risk_score = format_risk_report = trailing_stop_check = max_drawdown = sharpe_ratio = annualized_volatility = None

try:
    from backtest_engine import backtest as _backtest, edge as _edge, BacktestResult
except ImportError:
    _backtest = _edge = BacktestResult = None


# ---------------------------------------------------------------------------
# Data fetching helpers (reuse Stratapro's multi-source approach)
# ---------------------------------------------------------------------------
import requests
import re

_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0 (compatible; Stratapro-MCP/4.0)",
}


def _sina_realtime(code: str) -> Optional[Dict[str, Any]]:
    """Fetch realtime quote from Sina Finance."""
    url = f"https://hq.sinajs.cn/list={code}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        r.encoding = "gbk"
        m = re.search(r'"([^"]+)"', r.text)
        if m:
            parts = m.group(1).split(",")
            if len(parts) >= 4:
                price = float(parts[3])
                prev = float(parts[2])
                pct = (price - prev) / prev * 100 if prev > 0 else 0
                return {
                    "name": parts[0],
                    "code": code,
                    "price": round(price, 2),
                    "prev_close": round(prev, 2),
                    "change_pct": round(pct, 2),
                    "open": float(parts[1]) if parts[1] else price,
                    "high": float(parts[4]) if len(parts) > 4 and parts[4] else price,
                    "low": float(parts[5]) if len(parts) > 5 and parts[5] else price,
                    "volume": float(parts[8]) if len(parts) > 8 and parts[8] else 0,
                    "timestamp": datetime.now().isoformat(),
                }
    except Exception:
        pass
    return None


def _sina_kline(code: str, count: int = 120) -> List[Dict[str, Any]]:
    """Fetch daily K-line from Sina Finance."""
    url = (
        f"https://quotes.sina.cn/cn/api/json_v2.php/"
        f"CN_MarketDataService.getKLineData?symbol={code}&scale=240&datalen={count}&ma=no"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _tencent_realtime(code: str) -> Optional[Dict[str, Any]]:
    """Fallback: fetch realtime quote from Tencent Finance."""
    url = f"https://qt.gtimg.cn/q={code}"
    try:
        r = requests.get(url, timeout=10)
        r.encoding = "gbk"
        m = re.search(r'"([^"]+)"', r.text)
        if m:
            parts = m.group(1).split("~")
            if len(parts) >= 45:
                return {
                    "name": parts[1],
                    "code": code,
                    "price": float(parts[3]) if parts[3] else 0,
                    "change_pct": float(parts[32]) if parts[32] else 0,
                    "volume": float(parts[6]) if parts[6] else 0,
                    "source": "tencent",
                    "timestamp": datetime.now().isoformat(),
                }
    except Exception:
        pass
    return None


def _fetch_quote(code: str) -> Optional[Dict[str, Any]]:
    """Multi-source quote with fallback."""
    result = _sina_realtime(code)
    if result:
        return result
    return _tencent_realtime(code)


# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------
if _HAS_MCP:
    mcp = FastMCP(
        "stratapro",
        instructions=(
            "Stratapro AI Quantitative Trading tools for A-stock market. "
            "Provides realtime quotes, technical analysis, risk assessment, "
            "backtesting, and paper trading. "
            "Data sources: Sina Finance (primary), Tencent Finance (fallback). "
            "Market: A-stock (Shanghai/Shenzhen). "
            "DISCLAIMER: All outputs are for research only, not investment advice."
        ),
    )
else:
    mcp = None


# ───────────────────────────── Market Data Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def get_price(code: str) -> Any:
        """Get realtime price for an A-stock.

        Args:
            code: Stock code, e.g. "sz002384", "sh600105", "sh000001" (indices).
        """
        quote = _fetch_quote(code)
        if not quote:
            return {"error": True, "message": f"无法获取 {code} 的行情数据"}
        return quote

    @mcp.tool()
    def get_klines(code: str, count: int = 120) -> Any:
        """Get daily K-line data for an A-stock.

        Args:
            code: Stock code, e.g. "sz002384".
            count: Number of bars, 1-500, default 120.
        """
        count = max(1, min(500, int(count)))
        data = _sina_kline(code, count)
        if not data:
            return {"error": True, "message": f"无法获取 {code} 的K线数据"}
        return {"code": code, "bars": len(data), "data": data}

    @mcp.tool()
    def get_index_quotes() -> Any:
        """Get major A-stock index quotes (Shanghai, Shenzhen, ChiNext, STAR50)."""
        indices = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sh000688": "科创50",
        }
        results = {}
        for code, name in indices.items():
            q = _fetch_quote(code)
            if q:
                results[code] = q
            else:
                results[code] = {"code": code, "name": name, "error": True}
        return results


# ───────────────────────────── Analysis Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def analyze_stock(code: str) -> Any:
        """Full technical analysis for a single A-stock.

        Returns: realtime quote, RSI, MACD, KDJ, Bollinger, ATR, momentum score,
        risk score, and dynamic-weight composite score.
        """
        quote = _fetch_quote(code)
        if not quote:
            return {"error": True, "message": f"无法获取 {code} 数据"}

        kdata = _sina_kline(code, 120)
        closes = [float(d.get("close", 0)) for d in kdata if d.get("close")]
        highs = [float(d.get("high", 0)) for d in kdata if d.get("high")]
        lows = [float(d.get("low", 0)) for d in kdata if d.get("low")]

        result: Dict[str, Any] = {"quote": quote}

        if len(closes) >= 14 and compute_momentum_score:
            momentum = compute_momentum_score(closes, highs, lows)
            result["momentum"] = momentum

        if len(closes) >= 20 and calculate_risk_score:
            risk = calculate_risk_score(closes)
            result["risk"] = risk

        # Compute individual indicators
        indicators: Dict[str, Any] = {}
        if closes and len(closes) >= 14 and compute_rsi:
            indicators["rsi_14"] = compute_rsi(closes)
        if closes and len(closes) >= 35 and compute_macd:
            dif, dea, bar = compute_macd(closes)
            indicators["macd"] = {"dif": dif, "dea": dea, "bar": bar}
        if highs and lows and closes and len(closes) >= 9 and compute_kdj:
            k, d, j = compute_kdj(highs, lows, closes)
            indicators["kdj"] = {"k": k, "d": d, "j": j}
        if closes and len(closes) >= 20 and compute_bollinger:
            upper, mid, lower, pct_b = compute_bollinger(closes)
            indicators["bollinger"] = {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b}
        if highs and lows and closes and len(closes) >= 15 and compute_atr:
            indicators["atr_14"] = compute_atr(highs, lows, closes)

        result["indicators"] = indicators

        # Market state detection
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            cur = closes[-1]
            above = cur > ma20
            recent_5 = sum(closes[-5:]) / 5
            older_15 = sum(closes[-20:-5]) / 15
            slope = recent_5 > older_15
            if above and slope:
                result["market_state"] = "bull"
                result["weights"] = {"technical": 0.60, "fundamental": 0.25, "industry": 0.15}
            elif not above and not slope:
                result["market_state"] = "bear"
                result["weights"] = {"technical": 0.25, "fundamental": 0.50, "industry": 0.25}
            else:
                result["market_state"] = "mixed"
                result["weights"] = {"technical": 0.40, "fundamental": 0.35, "industry": 0.25}

        return result


# ───────────────────────────── Risk Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def assess_risk(code: str, entry_price: float = 0.0) -> Any:
        """Assess risk for an A-stock position.

        Args:
            code: Stock code, e.g. "sz002384".
            entry_price: Your entry price (0 = skip stop-loss check).
        """
        kdata = _sina_kline(code, 120)
        closes = [float(d.get("close", 0)) for d in kdata if d.get("close")]
        if len(closes) < 20:
            return {"error": True, "message": f"{code} K线数据不足"}

        if calculate_risk_score:
            ep = entry_price if entry_price > 0 else None
            risk = calculate_risk_score(closes, entry_price=ep)
            return risk
        return {"error": True, "message": "风险模块未加载"}

    @mcp.tool()
    def check_stop_loss(code: str, entry_price: float, stop_pct: float = 8.0, take_profit_pct: float = 20.0) -> Any:
        """Check trailing stop-loss / take-profit for a position.

        Args:
            code: Stock code.
            entry_price: Your entry price.
            stop_pct: Stop-loss percentage (default 8%).
            take_profit_pct: Take-profit percentage (default 20%).
        """
        quote = _fetch_quote(code)
        if not quote:
            return {"error": True, "message": f"无法获取 {code} 行情"}

        current_price = quote.get("price", 0)
        if current_price <= 0:
            return {"error": True, "message": f"{code} 价格异常"}

        if trailing_stop_check:
            result = trailing_stop_check(current_price, entry_price, stop_pct, take_profit_pct)
            result["stock"] = quote.get("name", code)
            result["code"] = code
            return result
        return {"error": True, "message": "止损模块未加载"}


# ───────────────────────────── Factor Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def compute_factors(code: str) -> Any:
        """Compute all technical factors (RSI, MACD, KDJ, Bollinger, ATR, momentum) for a stock.

        Args:
            code: Stock code, e.g. "sz002384".
        """
        kdata = _sina_kline(code, 120)
        closes = [float(d.get("close", 0)) for d in kdata if d.get("close")]
        highs = [float(d.get("high", 0)) for d in kdata if d.get("high")]
        lows = [float(d.get("low", 0)) for d in kdata if d.get("low")]

        if len(closes) < 14:
            return {"error": True, "message": f"{code} 数据不足"}

        factors: Dict[str, Any] = {}
        if compute_rsi:
            factors["rsi_14"] = compute_rsi(closes)
        if compute_macd and len(closes) >= 35:
            dif, dea, bar = compute_macd(closes)
            factors["macd"] = {"dif": dif, "dea": dea, "bar": bar}
        if compute_kdj and highs and lows:
            k, d, j = compute_kdj(highs, lows, closes)
            factors["kdj"] = {"k": k, "d": d, "j": j}
        if compute_bollinger and len(closes) >= 20:
            upper, mid, lower, pct_b = compute_bollinger(closes)
            factors["bollinger"] = {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b}
        if compute_atr and highs and lows and len(closes) >= 15:
            factors["atr_14"] = compute_atr(highs, lows, closes)
        if compute_momentum_score and highs and lows:
            factors["momentum"] = compute_momentum_score(closes, highs, lows)

        return {"code": code, "bars": len(closes), "factors": factors}


# ───────────────────────────── Backtest Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def run_backtest(
        code: str,
        strategy: str = "dual_ma",
        initial_capital: float = 100000.0,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
        position_size_pct: float = 0.25,
        short_window: int = 5,
        long_window: int = 20,
        kline_count: int = 250,
    ) -> Any:
        """Run a backtest on an A-stock with built-in strategies.

        Args:
            code: Stock code, e.g. "sz002384".
            strategy: Strategy name — "dual_ma" (dual moving average crossover), "rsi" (RSI overbought/oversold).
            initial_capital: Starting capital in CNY (default 100,000).
            stop_loss_pct: Engine stop-loss (default 8%).
            take_profit_pct: Engine take-profit (default 20%).
            position_size_pct: Position size fraction (default 25%).
            short_window: Short MA period (default 5).
            long_window: Long MA period (default 20).
            kline_count: Number of K-line bars (default 250).
        """
        if not _backtest:
            return {"error": True, "message": "回测引擎未加载"}

        kdata = _sina_kline(code, kline_count)
        closes = [float(d.get("close", 0)) for d in kdata if d.get("close")]
        if len(closes) < max(short_window, long_window) + 5:
            return {"error": True, "message": f"{code} K线数据不足"}

        signals = []

        if strategy == "dual_ma":
            # Dual MA crossover
            short_ma = [sum(closes[max(0, i - short_window + 1):i + 1]) / min(i + 1, short_window) for i in range(len(closes))]
            long_ma = [sum(closes[max(0, i - long_window + 1):i + 1]) / min(i + 1, long_window) for i in range(len(closes))]
            golden = _edge(short_ma[i] > long_ma[i] for i in range(len(closes)))
            death = _edge(short_ma[i] < long_ma[i] for i in range(len(closes)))
            for i in range(len(closes)):
                if golden[i]:
                    signals.append({"index": i, "action": "open_long", "reason": f"金叉 MA{short_window}>MA{long_window}"})
                elif death[i]:
                    signals.append({"index": i, "action": "close_long", "reason": f"死叉 MA{short_window}<MA{long_window}"})

        elif strategy == "rsi":
            # RSI strategy: buy when RSI < 30, sell when RSI > 70
            for i in range(15, len(closes)):
                rsi = compute_rsi(closes[:i + 1]) if compute_rsi else None
                if rsi is not None:
                    if rsi < 30:
                        signals.append({"index": i, "action": "open_long", "reason": f"RSI超卖 {rsi:.1f}"})
                    elif rsi > 70:
                        signals.append({"index": i, "action": "close_long", "reason": f"RSI超买 {rsi:.1f}"})
        else:
            return {"error": True, "message": f"未知策略: {strategy}。支持: dual_ma, rsi"}

        dates = [d.get("day", d.get("date", f"day_{i}")) for i, d in enumerate(kdata[:len(closes)])]
        result = _backtest(
            prices=closes,
            signals=signals,
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            position_size_pct=position_size_pct,
            strategy_name=f"{strategy}_{code}",
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
        )

        return {
            "strategy": strategy,
            "code": code,
            "summary": result.summary(),
            "metrics": {
                "total_return_pct": result.total_return,
                "max_drawdown_pct": result.max_drawdown,
                "win_rate_pct": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "initial_capital": result.initial_capital,
                "final_capital": result.final_capital,
            },
        }


# ───────────────────────────── Paper Trading Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def get_portfolio() -> Any:
        """Get current paper trading portfolio (positions and cash)."""
        from paper_trading import PaperTradingEngine
        engine = PaperTradingEngine()
        return engine.get_portfolio()

    @mcp.tool()
    def paper_buy(code: str, amount: float = 0, shares: int = 0, reason: str = "") -> Any:
        """Place a paper BUY order.

        Args:
            code: Stock code, e.g. "sz002384".
            amount: CNY amount to invest (0 = use shares instead).
            shares: Number of shares (used if amount=0).
            reason: Trade reason for audit log.
        """
        from paper_trading import PaperTradingEngine
        engine = PaperTradingEngine()
        if amount > 0:
            return engine.buy_by_amount(code, amount, reason=reason)
        elif shares > 0:
            return engine.buy_by_shares(code, shares, reason=reason)
        return {"error": True, "message": "请指定 amount (金额) 或 shares (股数)"}

    @mcp.tool()
    def paper_sell(code: str, shares: int = 0, sell_all: bool = False, reason: str = "") -> Any:
        """Place a paper SELL order.

        Args:
            code: Stock code, e.g. "sz002384".
            shares: Number of shares to sell (0 = ignored if sell_all=True).
            sell_all: If True, sell entire position.
            reason: Trade reason for audit log.
        """
        from paper_trading import PaperTradingEngine
        engine = PaperTradingEngine()
        if sell_all:
            return engine.sell_all(code, reason=reason)
        elif shares > 0:
            return engine.sell_by_shares(code, shares, reason=reason)
        return {"error": True, "message": "请指定 shares (股数) 或 sell_all=True"}

    @mcp.tool()
    def get_paper_history(limit: int = 50) -> Any:
        """Get paper trading order history.

        Args:
            limit: Max orders to return (1-200, default 50).
        """
        from paper_trading import PaperTradingEngine
        engine = PaperTradingEngine()
        return engine.get_history(limit=max(1, min(200, int(limit))))

    @mcp.tool()
    def reset_paper_account(initial_capital: float = 1000000.0) -> Any:
        """Reset paper trading account to initial state.

        Args:
            initial_capital: Starting capital in CNY (default 1,000,000).
        """
        from paper_trading import PaperTradingEngine
        engine = PaperTradingEngine(initial_capital=initial_capital)
        engine.reset()
        return {"message": f"模拟账户已重置，初始资金: ¥{initial_capital:,.0f}"}


# ───────────────────────────── Watchlist Tools ─────────────────────────────

if _HAS_MCP:
    @mcp.tool()
    def get_watchlist() -> Any:
        """Get Stratapro's default watchlist (持仓 + 关注 stocks)."""
        portfolio = {
            "positions": [
                {"code": "sz002384", "name": "东山精密", "sector": "光通信/EML"},
                {"code": "sh600105", "name": "永鼎股份", "sector": "CW光源"},
                {"code": "sh600576", "name": "汇金银行", "sector": "银行"},
                {"code": "sz300124", "name": "汇川技术", "sector": "机器人伺服"},
                {"code": "sh688017", "name": "绿的谐波", "sector": "谐波减速器"},
            ],
            "watchlist": [
                "sh600487", "sz300308", "sz002156", "sh600584",
                "sh600522", "sz002025", "sz000001",
            ],
        }
        # Enrich with realtime prices
        for item in portfolio["positions"]:
            q = _fetch_quote(item["code"])
            if q:
                item["price"] = q.get("price", 0)
                item["change_pct"] = q.get("change_pct", 0)

        return portfolio

    @mcp.tool()
    def scan_market() -> Any:
        """Quick market scan: index quotes + portfolio snapshot + risk alerts."""
        result: Dict[str, Any] = {}

        # Index quotes
        indices = {"sh000001": "上证指数", "sz399001": "深证成指", "sz399006": "创业板指"}
        result["indices"] = {}
        for code, name in indices.items():
            q = _fetch_quote(code)
            if q:
                result["indices"][code] = {"name": name, "price": q["price"], "change_pct": q["change_pct"]}

        # Portfolio positions
        holdings = ["sz002384", "sh600105", "sh600576", "sz300124", "sh688017"]
        result["portfolio"] = []
        for code in holdings:
            q = _fetch_quote(code)
            if q:
                result["portfolio"].append({
                    "code": code,
                    "name": q.get("name", ""),
                    "price": q.get("price", 0),
                    "change_pct": q.get("change_pct", 0),
                })

        # Risk alerts
        alerts = []
        for pos in result["portfolio"]:
            if pos["change_pct"] <= -5:
                alerts.append({"level": "critical", "stock": pos["name"], "msg": f"跌幅{pos['change_pct']:.1f}%，触发持仓警戒"})
            elif pos["change_pct"] >= 9.8:
                alerts.append({"level": "info", "stock": pos["name"], "msg": f"涨幅{pos['change_pct']:.1f}%，可能涨停"})
            elif pos["change_pct"] <= -2:
                alerts.append({"level": "warning", "stock": pos["name"], "msg": f"跌幅{pos['change_pct']:.1f}%，注意风险"})
        result["alerts"] = alerts

        result["timestamp"] = datetime.now().isoformat()
        result["disclaimer"] = "仅供参考，不构成投资建议"
        return result


# ───────────────────────────── Server Entrypoint ─────────────────────────────

def main():
    """Run the Stratapro MCP server."""
    if not _HAS_MCP:
        print(
            "[stratapro-mcp] Error: 'mcp' package not installed.\n"
            "  Install: pip install mcp\n"
            "  Or: pip install 'mcp[cli]'",
            file=sys.stderr,
        )
        sys.exit(1)

    transport = os.environ.get("STRATAPRO_MCP_TRANSPORT", "stdio").strip().lower()
    if transport in ("http", "streamable-http", "streamable_http"):
        transport = "streamable-http"

    host = os.environ.get("STRATAPRO_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("STRATAPRO_MCP_PORT", "7900"))

    if transport in ("sse", "streamable-http"):
        settings = getattr(mcp, "settings", None)
        if settings:
            try:
                settings.host = host
                settings.port = port
            except Exception:
                pass

    print(f"[stratapro-mcp] Starting MCP server (transport={transport})", file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
