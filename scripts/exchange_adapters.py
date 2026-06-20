# -*- coding: utf-8 -*-
"""
Exchange Adapters v4.0 — Stratapro 多交易所/多市场适配层
========================================================
Unified interface for multiple data sources and trading venues.

Supported adapters:
  - AStockSina:      A-stock via Sina Finance (default)
  - AStockTencent:   A-stock via Tencent Finance (fallback)
  - CryptoCCXT:      Crypto via CCXT library (Binance, OKX, Bybit, etc.)
  - UniversalAdapter: Abstract base for custom venue integration

Inspired by QuantDinger's multi-venue architecture.

Author: AtomCollide-智械工坊团队
Version: 4.0
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Common data types
# ---------------------------------------------------------------------------

@dataclass
class Ticker:
    """Unified ticker across all venues."""
    symbol: str
    venue: str               # "astock", "binance", "okx", etc.
    price: float
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0
    change_pct: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    timestamp: str = ""
    raw: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "venue": self.venue,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "volume_24h": self.volume_24h,
            "change_pct": self.change_pct,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "timestamp": self.timestamp,
        }


@dataclass
class OHLCV:
    """Unified OHLCV bar."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class OrderRequest:
    """Unified order request for all venues."""
    symbol: str
    side: str            # "buy" / "sell"
    quantity: float      # shares or base currency amount
    order_type: str = "market"   # "market" / "limit"
    limit_price: float = 0.0
    venue: str = "paper"
    reason: str = ""


@dataclass
class OrderResult:
    """Unified order result."""
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    filled_quantity: float = 0.0
    commission: float = 0.0
    message: str = ""
    venue: str = ""


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------

class ExchangeAdapter(ABC):
    """Base class for all exchange/venue adapters."""

    VENUE_NAME: str = "unknown"

    @abstractmethod
    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get realtime ticker."""
        ...

    @abstractmethod
    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[OHLCV]:
        """Get OHLCV bars."""
        ...

    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[Dict[str, Any]]:
        """Get order book (optional, not all venues support)."""
        return None

    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place order (optional, requires trading permission)."""
        return OrderResult(success=False, message=f"{self.VENUE_NAME} 不支持下单")

    def get_balance(self) -> Dict[str, float]:
        """Get account balance (optional)."""
        return {}

    def supports_trading(self) -> bool:
        """Whether this adapter supports order execution."""
        return False


# ---------------------------------------------------------------------------
# A-Stock: Sina Finance adapter
# ---------------------------------------------------------------------------

class AStockSina(ExchangeAdapter):
    """A-stock data via Sina Finance hq.sinajs.cn + quotes.sina.cn."""

    VENUE_NAME = "astock_sina"

    _HEADERS = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0 (compatible; Stratapro/4.0)",
    }

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        url = f"https://hq.sinajs.cn/list={symbol}"
        try:
            r = requests.get(url, headers=self._HEADERS, timeout=10)
            r.encoding = "gbk"
            m = re.search(r'"([^"]+)"', r.text)
            if m:
                parts = m.group(1).split(",")
                if len(parts) >= 9:
                    price = float(parts[3]) if parts[3] else 0
                    prev = float(parts[2]) if parts[2] else 0
                    pct = (price - prev) / prev * 100 if prev > 0 else 0
                    return Ticker(
                        symbol=symbol,
                        venue=self.VENUE_NAME,
                        price=price,
                        volume_24h=float(parts[8]) if parts[8] else 0,
                        change_pct=round(pct, 2),
                        high_24h=float(parts[4]) if parts[4] else 0,
                        low_24h=float(parts[5]) if parts[5] else 0,
                        timestamp=datetime.now().isoformat(),
                    )
        except Exception:
            pass
        return None

    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[OHLCV]:
        # Sina K-line: scale=240=daily, scale=60=hourly, scale=15=15min
        scale_map = {"1d": 240, "1h": 60, "15m": 15, "5m": 5, "1m": 1}
        scale = scale_map.get(timeframe, 240)
        url = (
            f"https://quotes.sina.cn/cn/api/json_v2.php/"
            f"CN_MarketDataService.getKLineData?symbol={symbol}&scale={scale}&datalen={limit}&ma=no"
        )
        try:
            r = requests.get(url, headers=self._HEADERS, timeout=15)
            data = r.json()
            if isinstance(data, list):
                result = []
                for d in data:
                    result.append(OHLCV(
                        timestamp=d.get("day", d.get("date", "")),
                        open=float(d.get("open", 0)),
                        high=float(d.get("high", 0)),
                        low=float(d.get("low", 0)),
                        close=float(d.get("close", 0)),
                        volume=float(d.get("volume", 0)),
                    ))
                return result
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# A-Stock: Tencent Finance adapter (fallback)
# ---------------------------------------------------------------------------

class AStockTencent(ExchangeAdapter):
    """A-stock data via Tencent Finance qt.gtimg.cn."""

    VENUE_NAME = "astock_tencent"

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        url = f"https://qt.gtimg.cn/q={symbol}"
        try:
            r = requests.get(url, timeout=10)
            r.encoding = "gbk"
            m = re.search(r'"([^"]+)"', r.text)
            if m:
                parts = m.group(1).split("~")
                if len(parts) >= 45:
                    return Ticker(
                        symbol=symbol,
                        venue=self.VENUE_NAME,
                        price=float(parts[3]) if parts[3] else 0,
                        volume_24h=float(parts[6]) if parts[6] else 0,
                        change_pct=float(parts[32]) if parts[32] else 0,
                        high_24h=float(parts[33]) if len(parts) > 33 and parts[33] else 0,
                        low_24h=float(parts[34]) if len(parts) > 34 and parts[34] else 0,
                        timestamp=datetime.now().isoformat(),
                    )
        except Exception:
            pass
        return None

    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[OHLCV]:
        # Tencent uses web.ifzq.gtimg.cn for K-lines
        period_map = {"1d": "day", "1h": "60", "15m": "15", "5m": "5"}
        period = period_map.get(timeframe, "day")
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},{period},,,{limit},qfq"
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            bars = []
            if isinstance(data, dict):
                code_data = data.get("data", {}).get(symbol, {})
                kline_key = f"qfq{period}" if f"qfq{period}" in code_data else period
                raw = code_data.get(kline_key, [])
                for item in raw:
                    if len(item) >= 6:
                        bars.append(OHLCV(
                            timestamp=item[0],
                            open=float(item[1]),
                            close=float(item[2]),
                            high=float(item[3]),
                            low=float(item[4]),
                            volume=float(item[5]) if len(item) > 5 else 0,
                        ))
            return bars
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Crypto: CCXT adapter (Binance, OKX, Bybit, etc.)
# ---------------------------------------------------------------------------

class CryptoCCXT(ExchangeAdapter):
    """
    Crypto exchange adapter via CCXT library.
    
    Requires: pip install ccxt
    
    Supported exchanges: binance, okx, bybit, bitget, gate, huobi, kucoin, etc.
    """

    VENUE_NAME = "crypto_ccxt"

    def __init__(self, exchange_id: str = "binance", api_key: str = "", secret: str = "", passphrase: str = ""):
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._secret = secret
        self._passphrase = passphrase
        self._exchange = None
        self._init_ccxt()

    def _init_ccxt(self):
        try:
            import ccxt
            exchange_class = getattr(ccxt, self._exchange_id, None)
            if not exchange_class:
                raise ValueError(f"不支持的交易所: {self._exchange_id}")
            config = {"apiKey": self._api_key, "secret": self._secret, "timeout": 30000, "enableRateLimit": True}
            if self._passphrase:
                config["password"] = self._passphrase
            # If no API key, use public-only mode
            if not self._api_key:
                config.pop("apiKey")
                config.pop("secret")
            self._exchange = exchange_class(config)
        except ImportError:
            self._exchange = None
        except Exception:
            self._exchange = None

    @property
    def is_available(self) -> bool:
        return self._exchange is not None

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        if not self._exchange:
            return None
        try:
            t = self._exchange.fetch_ticker(symbol)
            return Ticker(
                symbol=symbol,
                venue=f"crypto_{self._exchange_id}",
                price=t.get("last", 0),
                bid=t.get("bid", 0),
                ask=t.get("ask", 0),
                volume_24h=t.get("baseVolume", 0),
                change_pct=round(t.get("percentage", 0), 2),
                high_24h=t.get("high", 0),
                low_24h=t.get("low", 0),
                timestamp=datetime.fromtimestamp(t.get("timestamp", 0) / 1000).isoformat() if t.get("timestamp") else "",
            )
        except Exception:
            return None

    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> List[OHLCV]:
        if not self._exchange:
            return []
        try:
            bars = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            result = []
            for bar in bars:
                if len(bar) >= 6:
                    result.append(OHLCV(
                        timestamp=datetime.fromtimestamp(bar[0] / 1000).isoformat(),
                        open=float(bar[1]),
                        high=float(bar[2]),
                        low=float(bar[3]),
                        close=float(bar[4]),
                        volume=float(bar[5]),
                    ))
            return result
        except Exception:
            return []

    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[Dict[str, Any]]:
        if not self._exchange:
            return None
        try:
            ob = self._exchange.fetch_order_book(symbol, limit=depth)
            return {
                "symbol": symbol,
                "venue": f"crypto_{self._exchange_id}",
                "bids": ob.get("bids", [])[:depth],
                "asks": ob.get("asks", [])[:depth],
                "timestamp": ob.get("datetime", ""),
            }
        except Exception:
            return None

    def place_order(self, order: OrderRequest) -> OrderResult:
        if not self._exchange:
            return OrderResult(success=False, message="CCXT 未初始化")
        if not self._api_key:
            return OrderResult(success=False, message="未配置 API Key，仅支持行情查询")

        try:
            if order.order_type == "market":
                result = self._exchange.create_order(
                    symbol=order.symbol,
                    type="market",
                    side=order.side,
                    amount=order.quantity,
                )
            else:
                result = self._exchange.create_order(
                    symbol=order.symbol,
                    type="limit",
                    side=order.side,
                    amount=order.quantity,
                    price=order.limit_price,
                )
            return OrderResult(
                success=True,
                order_id=result.get("id", ""),
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=result.get("average", result.get("price", 0)),
                filled_quantity=result.get("filled", order.quantity),
                venue=f"crypto_{self._exchange_id}",
            )
        except Exception as e:
            return OrderResult(success=False, message=str(e))

    def get_balance(self) -> Dict[str, float]:
        if not self._exchange or not self._api_key:
            return {}
        try:
            bal = self._exchange.fetch_balance()
            result = {}
            for currency, info in bal.get("total", {}).items():
                if info and float(info) > 0:
                    result[currency] = float(info)
            return result
        except Exception:
            return {}

    def supports_trading(self) -> bool:
        return bool(self._api_key)


# ---------------------------------------------------------------------------
# Multi-venue aggregator
# ---------------------------------------------------------------------------

class MultiVenueManager:
    """
    Manages multiple exchange adapters with automatic fallback.
    
    Usage:
        manager = MultiVenueManager()
        manager.add_adapter(AStockSina())
        manager.add_adapter(AStockTencent())
        ticker = manager.get_ticker("sz002384")
    """

    def __init__(self):
        self._adapters: List[ExchangeAdapter] = []
        self._crypto_adapters: Dict[str, CryptoCCXT] = {}

    def add_adapter(self, adapter: ExchangeAdapter) -> None:
        self._adapters.append(adapter)

    def add_crypto_exchange(self, exchange_id: str, api_key: str = "", secret: str = "", passphrase: str = "") -> CryptoCCXT:
        """Add a crypto exchange via CCXT."""
        adapter = CryptoCCXT(exchange_id, api_key, secret, passphrase)
        self._crypto_adapters[exchange_id] = adapter
        self._adapters.append(adapter)
        return adapter

    def get_ticker(self, symbol: str, venue: str = "") -> Optional[Ticker]:
        """Get ticker with automatic fallback across all adapters."""
        if venue:
            # Try specific venue
            for adapter in self._adapters:
                if adapter.VENUE_NAME == venue or adapter.VENUE_NAME.endswith(venue):
                    result = adapter.get_ticker(symbol)
                    if result:
                        return result
            return None

        # Try all adapters in order
        for adapter in self._adapters:
            result = adapter.get_ticker(symbol)
            if result:
                return result
        return None

    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 100, venue: str = "") -> List[OHLCV]:
        """Get klines with automatic fallback."""
        if venue:
            for adapter in self._adapters:
                if adapter.VENUE_NAME == venue or adapter.VENUE_NAME.endswith(venue):
                    return adapter.get_klines(symbol, timeframe, limit)

        for adapter in self._adapters:
            result = adapter.get_klines(symbol, timeframe, limit)
            if result:
                return result
        return []

    def get_multi_venue_tickers(self, symbol: str) -> Dict[str, Ticker]:
        """Get ticker from all available venues."""
        results = {}
        for adapter in self._adapters:
            ticker = adapter.get_ticker(symbol)
            if ticker:
                results[adapter.VENUE_NAME] = ticker
        return results

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters."""
        out = []
        for adapter in self._adapters:
            out.append({
                "venue": adapter.VENUE_NAME,
                "trading": adapter.supports_trading(),
                "type": "crypto" if isinstance(adapter, CryptoCCXT) else "astock",
            })
        return out

    @property
    def available_venues(self) -> List[str]:
        return [a.VENUE_NAME for a in self._adapters]


# ---------------------------------------------------------------------------
# Default instance factory
# ---------------------------------------------------------------------------

def create_default_manager(include_crypto: bool = False, crypto_exchanges: Optional[List[str]] = None) -> MultiVenueManager:
    """Create a MultiVenueManager with default A-stock adapters."""
    import os

    manager = MultiVenueManager()

    # A-stock adapters (always available, no API key needed)
    manager.add_adapter(AStockSina())
    manager.add_adapter(AStockTencent())

    # Crypto adapters (optional, requires ccxt + API keys)
    if include_crypto:
        exchanges = crypto_exchanges or ["binance", "okx"]
        for ex_id in exchanges:
            api_key = os.environ.get(f"{ex_id.upper()}_API_KEY", "")
            secret = os.environ.get(f"{ex_id.upper()}_SECRET", "")
            passphrase = os.environ.get(f"{ex_id.upper()}_PASSPHRASE", "")
            manager.add_crypto_exchange(ex_id, api_key, secret, passphrase)

    return manager


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    import sys
    import json as _json

    if len(sys.argv) < 2:
        print("Stratapro 多市场适配器")
        print("用法:")
        print("  python exchange_adapters.py quote <代码>        — 获取行情")
        print("  python exchange_adapters.py kline <代码> [天数]  — 获取K线")
        print("  python exchange_adapters.py venues              — 列出适配器")
        print("  python exchange_adapters.py crypto <交易所> <交易对> — 加密货币行情")
        print("")
        print("示例:")
        print("  python exchange_adapters.py quote sz002384")
        print("  python exchange_adapters.py crypto binance BTC/USDT")
        return

    cmd = sys.argv[1]
    manager = create_default_manager(include_crypto=True)

    if cmd == "quote" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        ticker = manager.get_ticker(symbol)
        if ticker:
            print(f"{ticker.symbol} @ {ticker.venue}")
            print(f"  价格: {ticker.price}")
            print(f"  涨跌: {ticker.change_pct:+.2f}%")
            print(f"  最高: {ticker.high_24h}  最低: {ticker.low_24h}")
            print(f"  成交量: {ticker.volume_24h:,.0f}")
        else:
            print(f"❌ 无法获取 {symbol} 行情")

    elif cmd == "kline" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        bars = manager.get_klines(symbol, limit=limit)
        print(f"最近 {len(bars)} 根K线:")
        for bar in bars[-10:]:
            print(f"  {bar.timestamp}  O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:,.0f}")

    elif cmd == "venues":
        adapters = manager.list_adapters()
        for a in adapters:
            t = "✅" if a["trading"] else "📊"
            print(f"  {t} {a['venue']} ({a['type']})")

    elif cmd == "crypto" and len(sys.argv) >= 4:
        exchange = sys.argv[2]
        symbol = sys.argv[3]
        ticker = manager.get_ticker(symbol, venue=f"crypto_{exchange}")
        if ticker:
            print(f"{ticker.symbol} @ {exchange}")
            print(f"  价格: {ticker.price}")
            print(f"  涨跌: {ticker.change_pct:+.2f}%")
            print(f"  Bid: {ticker.bid}  Ask: {ticker.ask}")
        else:
            print(f"❌ 无法获取 {symbol}@{exchange} 行情 (请确认 ccxt 已安装)")

    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    main()
