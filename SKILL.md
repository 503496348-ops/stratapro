---
name: stratapro
description: "量化投资工具 — 市场报告、技术因子、模拟交易、回测引擎"
license: MIT
metadata:
  author: 503496348-ops
  version: 1.0.0
triggers:
  - "量化"
  - "股票"
  - "回测"
  - "技术因子"
  - "模拟交易"
  - "市场报告"
  - "quant"
---

# Stratapro — 量化投资工具

A 股量化分析全流程：市场状态检测、技术因子计算、模拟交易、策略回测。

## 核心能力

| 命令 | 说明 |
|------|------|
| `stratapro report` | 生成市场状态报告 |
| `stratapro factors` | 列出可用技术因子（RSI/MACD/KDJ/布林带等） |
| `stratapro trade status` | 查看模拟交易账户状态 |
| `stratapro backtest` | 运行策略回测 |

## 快速开始

```bash
# 生成市场报告
python3 scripts/cli.py report

# 查看可用因子
python3 scripts/cli.py factors

# 模拟交易
python3 scripts/cli.py trade buy --symbol 600519
python3 scripts/cli.py trade status
```

## 架构

- `scripts/market_report_v8.py` — 市场状态检测 + 腾讯数据源
- `scripts/factor_library.py` — 技术因子库（RSI/KDJ/MACD/ATR/布林带）
- `scripts/paper_trading.py` — 模拟交易引擎
- `scripts/backtest_engine.py` — 策略回测引擎
- `scripts/exchange_adapters.py` — 交易所适配器
- `scripts/mcp_server.py` — MCP 服务端

## 测试

```bash
python3 -m pytest tests/ -q
```
