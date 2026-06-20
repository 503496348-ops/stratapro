# -*- coding: utf-8 -*-
"""
Paper Trading Engine v4.0 — Stratapro 模拟交易引擎
===================================================
Simulated trading engine for A-stock paper trading.
Provides order management, portfolio tracking, and trade audit logging.

Features:
  - Buy/Sell by amount (CNY) or shares
  - Portfolio P&L tracking with realtime mark-to-market
  - Trade history with full audit trail
  - Position sizing and cash management
  - Integration with Stratapro risk module

Author: AtomCollide-智械工坊团队
Version: 4.0
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """A stock position in the paper portfolio."""
    code: str
    name: str
    shares: int
    avg_cost: float          # average cost per share
    entry_date: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def mark_to_market(self, price: float) -> None:
        self.current_price = round(price, 2)
        if self.shares > 0 and self.avg_cost > 0:
            self.unrealized_pnl = round((price - self.avg_cost) * self.shares, 2)
            self.unrealized_pnl_pct = round((price - self.avg_cost) / self.avg_cost * 100, 2)


@dataclass
class TradeRecord:
    """Audit log entry for every trade."""
    timestamp: str
    action: str              # buy / sell
    code: str
    name: str
    shares: int
    price: float
    amount: float            # total CNY
    reason: str = ""
    commission: float = 0.0


@dataclass
class PaperAccount:
    """Complete paper trading account state."""
    initial_capital: float = 1000000.0
    cash: float = 1000000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[TradeRecord] = field(default_factory=list)
    created_at: str = ""
    last_updated: str = ""


# ---------------------------------------------------------------------------
# Price fetching (reuse Stratapro multi-source)
# ---------------------------------------------------------------------------

_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0 (compatible; Stratapro-PaperTrade/4.0)",
}


def _sina_price(code: str) -> Optional[Dict[str, Any]]:
    url = f"https://hq.sinajs.cn/list={code}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        r.encoding = "gbk"
        m = re.search(r'"([^"]+)"', r.text)
        if m:
            parts = m.group(1).split(",")
            if len(parts) >= 4:
                return {
                    "name": parts[0],
                    "price": float(parts[3]) if parts[3] else 0,
                    "prev_close": float(parts[2]) if parts[2] else 0,
                }
    except Exception:
        pass
    return None


def _tencent_price(code: str) -> Optional[Dict[str, Any]]:
    url = f"https://qt.gtimg.cn/q={code}"
    try:
        r = requests.get(url, timeout=10)
        r.encoding = "gbk"
        m = re.search(r'"([^"]+)"', r.text)
        if m:
            parts = m.group(1).split("~")
            if len(parts) >= 5:
                return {
                    "name": parts[1],
                    "price": float(parts[3]) if parts[3] else 0,
                }
    except Exception:
        pass
    return None


def _fetch_price(code: str) -> Optional[Dict[str, Any]]:
    result = _sina_price(code)
    if result and result.get("price", 0) > 0:
        return result
    return _tencent_price(code)


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTradingEngine:
    """
    Simulated trading engine for A-stock paper trading.
    
    State is persisted to a JSON file in the project data directory.
    """

    DEFAULT_FILE = "paper_account.json"
    COMMISSION_RATE = 0.0003   # 万三佣金
    STAMP_TAX_RATE = 0.001     # 千一印花税（卖出时）
    MIN_COMMISSION = 5.0       # 最低佣金5元

    def __init__(self, initial_capital: float = 1000000.0, data_dir: str = ""):
        if not data_dir:
            # Auto-detect: scripts/ -> ../data/
            here = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(here), "data")
        os.makedirs(data_dir, exist_ok=True)
        self._state_file = os.path.join(data_dir, self.DEFAULT_FILE)
        self._initial_capital = initial_capital
        self._account = self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> PaperAccount:
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                account = PaperAccount(
                    initial_capital=data.get("initial_capital", self._initial_capital),
                    cash=data.get("cash", self._initial_capital),
                    created_at=data.get("created_at", ""),
                    last_updated=data.get("last_updated", ""),
                )
                for code, pos_data in data.get("positions", {}).items():
                    account.positions[code] = Position(**pos_data)
                for hist in data.get("history", []):
                    account.history.append(TradeRecord(**hist))
                return account
            except Exception:
                pass
        now = datetime.now().isoformat()
        return PaperAccount(
            initial_capital=self._initial_capital,
            cash=self._initial_capital,
            created_at=now,
            last_updated=now,
        )

    def _save(self) -> None:
        self._account.last_updated = datetime.now().isoformat()
        data = {
            "initial_capital": self._account.initial_capital,
            "cash": self._account.cash,
            "created_at": self._account.created_at,
            "last_updated": self._account.last_updated,
            "positions": {code: asdict(pos) for code, pos in self._account.positions.items()},
            "history": [asdict(h) for h in self._account.history[-500:]],  # keep last 500
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def reset(self) -> None:
        """Reset account to initial state."""
        now = datetime.now().isoformat()
        self._account = PaperAccount(
            initial_capital=self._initial_capital,
            cash=self._initial_capital,
            created_at=now,
            last_updated=now,
        )
        self._save()

    # ── Commission Calculation ───────────────────────────────────────────

    def _calc_commission(self, amount: float, is_sell: bool = False) -> float:
        """Calculate trading commission + stamp tax."""
        comm = max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
        if is_sell:
            comm += amount * self.STAMP_TAX_RATE  # 印花税只卖出收
        return round(comm, 2)

    # ── Buy Operations ───────────────────────────────────────────────────

    def buy_by_amount(self, code: str, amount: float, reason: str = "") -> Dict[str, Any]:
        """Buy stock by CNY amount. Automatically rounds to 100-share lots."""
        quote = _fetch_price(code)
        if not quote or quote.get("price", 0) <= 0:
            return {"error": True, "message": f"无法获取 {code} 实时价格"}

        price = quote["price"]
        name = quote.get("name", code)

        # A-stock: must buy in lots of 100
        shares = int(amount / price / 100) * 100
        if shares <= 0:
            return {"error": True, "message": f"金额 ¥{amount:,.0f} 不足以买入1手(100股)，当前价 ¥{price:.2f}"}

        total_amount = shares * price
        commission = self._calc_commission(total_amount, is_sell=False)
        total_cost = total_amount + commission

        if total_cost > self._account.cash:
            # Reduce shares to fit
            shares = int(self._account.cash / (price * (1 + self.COMMISSION_RATE)) / 100) * 100
            if shares <= 0:
                return {"error": True, "message": f"现金不足，当前可用: ¥{self._account.cash:,.2f}"}
            total_amount = shares * price
            commission = self._calc_commission(total_amount, is_sell=False)
            total_cost = total_amount + commission

        # Update position
        if code in self._account.positions:
            pos = self._account.positions[code]
            old_total = pos.avg_cost * pos.shares
            new_total = old_total + total_amount
            pos.shares += shares
            pos.avg_cost = round(new_total / pos.shares, 4)
        else:
            self._account.positions[code] = Position(
                code=code, name=name, shares=shares,
                avg_cost=round(price, 4), entry_date=datetime.now().strftime("%Y-%m-%d"),
                current_price=price,
            )

        self._account.cash -= total_cost

        # Audit log
        record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            action="buy", code=code, name=name,
            shares=shares, price=price, amount=total_amount,
            reason=reason, commission=commission,
        )
        self._account.history.append(record)
        self._save()

        return {
            "action": "buy",
            "code": code,
            "name": name,
            "shares": shares,
            "price": price,
            "amount": round(total_amount, 2),
            "commission": commission,
            "total_cost": round(total_cost, 2),
            "remaining_cash": round(self._account.cash, 2),
            "position_shares": self._account.positions[code].shares,
            "avg_cost": self._account.positions[code].avg_cost,
        }

    def buy_by_shares(self, code: str, shares: int, reason: str = "") -> Dict[str, Any]:
        """Buy stock by number of shares (rounded to 100-lot)."""
        shares = max(100, (shares // 100) * 100)  # round to lot
        quote = _fetch_price(code)
        if not quote or quote.get("price", 0) <= 0:
            return {"error": True, "message": f"无法获取 {code} 实时价格"}

        price = quote["price"]
        amount = shares * price
        return self.buy_by_amount(code, amount, reason=reason)

    # ── Sell Operations ──────────────────────────────────────────────────

    def sell_by_shares(self, code: str, shares: int, reason: str = "") -> Dict[str, Any]:
        """Sell stock by number of shares (rounded to 100-lot)."""
        if code not in self._account.positions:
            return {"error": True, "message": f"未持有 {code}"}

        pos = self._account.positions[code]
        shares = min(shares, pos.shares)
        shares = max(100, (shares // 100) * 100)
        if shares > pos.shares:
            shares = pos.shares

        quote = _fetch_price(code)
        if not quote or quote.get("price", 0) <= 0:
            return {"error": True, "message": f"无法获取 {code} 实时价格"}

        price = quote["price"]
        name = quote.get("name", code)
        total_amount = shares * price
        commission = self._calc_commission(total_amount, is_sell=True)
        net_amount = total_amount - commission

        # Update position
        pos.shares -= shares
        if pos.shares <= 0:
            del self._account.positions[code]

        self._account.cash += net_amount

        # P&L
        pnl = round((price - pos.avg_cost) * shares - commission, 2)
        pnl_pct = round((price - pos.avg_cost) / pos.avg_cost * 100, 2)

        # Audit log
        record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            action="sell", code=code, name=name,
            shares=shares, price=price, amount=total_amount,
            reason=reason, commission=commission,
        )
        self._account.history.append(record)
        self._save()

        return {
            "action": "sell",
            "code": code,
            "name": name,
            "shares": shares,
            "price": price,
            "amount": round(total_amount, 2),
            "commission": commission,
            "net_amount": round(net_amount, 2),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "remaining_cash": round(self._account.cash, 2),
        }

    def sell_all(self, code: str, reason: str = "") -> Dict[str, Any]:
        """Sell entire position."""
        if code not in self._account.positions:
            return {"error": True, "message": f"未持有 {code}"}
        return self.sell_by_shares(code, self._account.positions[code].shares, reason=reason)

    # ── Portfolio ────────────────────────────────────────────────────────

    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio with mark-to-market P&L."""
        total_position_value = 0.0
        positions_out = []

        for code, pos in self._account.positions.items():
            quote = _fetch_price(code)
            if quote and quote.get("price", 0) > 0:
                pos.mark_to_market(quote["price"])
            market_value = pos.current_price * pos.shares
            total_position_value += market_value
            positions_out.append(asdict(pos))

        total_value = self._account.cash + total_position_value
        total_pnl = total_value - self._account.initial_capital
        total_pnl_pct = (total_pnl / self._account.initial_capital * 100) if self._account.initial_capital > 0 else 0

        return {
            "cash": round(self._account.cash, 2),
            "position_value": round(total_position_value, 2),
            "total_value": round(total_value, 2),
            "initial_capital": self._account.initial_capital,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions": positions_out,
            "position_count": len(positions_out),
        }

    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get trade history."""
        history = self._account.history[-limit:]
        return {
            "total_trades": len(self._account.history),
            "showing": len(history),
            "history": [asdict(h) for h in reversed(history)],
        }


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    import sys
    engine = PaperTradingEngine()

    if len(sys.argv) < 2:
        print("用法:")
        print("  python paper_trading.py portfolio    — 查看持仓")
        print("  python paper_trading.py history      — 交易历史")
        print("  python paper_trading.py buy <代码> <金额>  — 买入")
        print("  python paper_trading.py sell <代码>        — 全部卖出")
        print("  python paper_trading.py reset             — 重置账户")
        print("")
        print("示例:")
        print("  python paper_trading.py buy sz002384 100000")
        print("  python paper_trading.py sell sz002384")
        print("  python paper_trading.py portfolio")
        return

    cmd = sys.argv[1]

    if cmd == "portfolio":
        result = engine.get_portfolio()
        print(f"═══ 模拟账户概况 ═══")
        print(f"总资金: ¥{result['total_value']:,.2f}")
        print(f"可用现金: ¥{result['cash']:,.2f}")
        print(f"持仓市值: ¥{result['position_value']:,.2f}")
        print(f"总盈亏: ¥{result['total_pnl']:,.2f} ({result['total_pnl_pct']:+.2f}%)")
        print(f"\n─── 持仓明细 ───")
        for pos in result["positions"]:
            print(f"  {pos['name']}({pos['code']}): {pos['shares']}股 "
                  f"成本{pos['avg_cost']:.2f} 现价{pos['current_price']:.2f} "
                  f"盈亏{pos['unrealized_pnl']:+.2f}({pos['unrealized_pnl_pct']:+.1f}%)")

    elif cmd == "history":
        result = engine.get_history()
        print(f"共 {result['total_trades']} 笔交易 (最近 {result['showing']} 笔)")
        for h in result["history"]:
            print(f"  [{h['timestamp'][:19]}] {h['action'].upper()} {h['name']} "
                  f"{h['shares']}股@¥{h['price']:.2f} ¥{h['amount']:,.2f} "
                  f"佣金¥{h['commission']:.2f} {h.get('reason', '')}")

    elif cmd == "buy" and len(sys.argv) >= 4:
        code = sys.argv[2]
        amount = float(sys.argv[3])
        result = engine.buy_by_amount(code, amount)
        if result.get("error"):
            print(f"❌ {result['message']}")
        else:
            print(f"✅ 买入 {result['name']}({result['code']}) {result['shares']}股 "
                  f"@¥{result['price']:.2f} 花费¥{result['total_cost']:,.2f}")

    elif cmd == "sell" and len(sys.argv) >= 3:
        code = sys.argv[2]
        result = engine.sell_all(code)
        if result.get("error"):
            print(f"❌ {result['message']}")
        else:
            print(f"✅ 卖出 {result['name']}({result['code']}) {result['shares']}股 "
                  f"@¥{result['price']:.2f} 盈亏¥{result['pnl']:+,.2f}({result['pnl_pct']:+.1f}%)")

    elif cmd == "reset":
        engine.reset()
        print("✅ 模拟账户已重置")

    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    main()
