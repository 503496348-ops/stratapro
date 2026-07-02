# Changelog

All notable changes to `stratapro` should be documented in this file.

This repository follows a lightweight Keep-a-Changelog style and semantic versioning where applicable.

## Unreleased

- Governance baseline initialized.

## v4.1 — 播客投资信号层

- 新增 `scripts/podcast_signal_extractor.py`：时间戳证据、字段模型、重复实体聚合、证据摘要。
- 新增 `tests/test_podcast_signal_extractor.py`：验证信号必须包含时间戳、原句和来源；三次可验证提及才形成趋势簇。
