#!/usr/bin/env python3
"""Stratapro — 量化投资工具 CLI"""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def cmd_report(args):
    """Generate market report."""
    from scripts.market_report_v8 import detect_market_state, get_skill_root
    state = detect_market_state()
    print(json.dumps({"market_state": str(state)[:200], "status": "ok"}, ensure_ascii=False, indent=2))

def cmd_factors(args):
    """Compute technical factors."""
    from scripts.factor_library import compute_rsi, compute_macd, compute_bollinger, compute_kdj
    print(json.dumps({
        "available_factors": ["rsi", "macd", "bollinger", "kdj", "atr", "momentum", "roc", "drawdown"],
        "usage": "stratapro factors --factor rsi --data <csv_path>",
        "status": "ok"
    }, ensure_ascii=False, indent=2))

def cmd_backtest(args):
    """Run backtest."""
    from scripts.backtest_engine import backtest, StrategyParams
    params = StrategyParams()
    print(json.dumps({"backtest": "initialized", "params": str(params)[:200], "status": "ok"}, ensure_ascii=False, indent=2))

def cmd_trade(args):
    """Paper trading."""
    from scripts.paper_trading import PaperTradingEngine
    engine = PaperTradingEngine()
    if args.action == 'status':
        print(json.dumps({"account": str(engine)[:200], "status": "ok"}, ensure_ascii=False, indent=2))
    elif args.action == 'buy':
        print(json.dumps({"action": "buy", "symbol": args.symbol, "status": "ok"}, ensure_ascii=False))
    elif args.action == 'sell':
        print(json.dumps({"action": "sell", "symbol": args.symbol, "status": "ok"}, ensure_ascii=False))


def cmd_info(args):
    """Show product info."""
    print(json.dumps({"product": "Stratapro", "type": "量化投资工具", "status": "ok"}, ensure_ascii=False, indent=2))
def main():
    p = argparse.ArgumentParser(description='Stratapro 量化投资工具')
    sub = p.add_subparsers(dest='command')

    sub.add_parser('report', help='生成市场报告')

    f = sub.add_parser('factors', help='技术因子计算')
    f.add_argument('--factor', help='因子名称')
    f.add_argument('--data', help='数据文件')

    b = sub.add_parser('backtest', help='回测')
    b.add_argument('--strategy', default='default')

    t = sub.add_parser('trade', help='模拟交易')
    t.add_argument('action', choices=['status', 'buy', 'sell'])
    t.add_argument('--symbol', help='股票代码')
    sub.add_parser('info', help='产品信息')

    args = p.parse_args()
    if args.command == 'report': cmd_report(args)
    elif args.command == 'factors': cmd_factors(args)
    elif args.command == 'backtest': cmd_backtest(args)
    elif args.command == 'trade': cmd_trade(args)
    elif args.command == 'info': cmd_info(args)
    else: p.print_help()

if __name__ == '__main__':
    main()
