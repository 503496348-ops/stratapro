# Changelog

All notable changes to `stratapro` should be documented in this file.

This repository follows a lightweight Keep-a-Changelog style and semantic versioning where applicable.

## [2.0.0] — 2026-07-04

### Added — Qlib Multi-Factor ML Enhancement
- **Enhanced factor library** (`scripts/factor_library.py`): 8 new Qlib-inspired alpha factors — multi-period ROC (5/10/20), max drawdown, volume-price correlation, volatility ratio, Williams %R, CCI, OBV score, and `enhanced_momentum_score` (24-dim combined score, 0-100).
- **ML ensemble scorer** (`scripts/ml_scorer.py`): Lightweight logistic regression with hardcoded Qlib Alpha158 coefficients. `build_feature_vector` → `ml_ensemble_score` → `score_stock` pipeline. Weights: momentum 30%, volatility 20%, volume-price 15%, trend 20%, mean-reversion 15%. No sklearn dependency.

### Changed
- `factor_library.py` now exports 14 factor functions (up from 6).

## Unreleased

- Governance baseline initialized.

## v4.1 — 播客投资信号层

- 新增 `scripts/podcast_signal_extractor.py`：时间戳证据、字段模型、重复实体聚合、证据摘要。
- 新增 `tests/test_podcast_signal_extractor.py`：验证信号必须包含时间戳、原句和来源；三次可验证提及才形成趋势簇。
