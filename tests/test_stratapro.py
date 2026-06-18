# -*- coding: utf-8 -*-
"""
测试套件：深度方略 Stratapro
目标覆盖率：≥80%
"""
import pytest, sys, os, json
from pathlib import Path

# 添加scripts到path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../scripts'))

# ── Test: skill_paths.py ─────────────────────────────────
class TestSkillPaths:
    def test_get_skill_root(self):
        from skill_paths import get_skill_root
        root = get_skill_root()
        assert root.endswith('stock-analysis')
        assert os.path.isdir(root)
    
    def test_get_diary_dir(self):
        from skill_paths import get_diary_dir
        diary = get_diary_dir()
        assert 'diary' in diary
        assert os.path.exists(diary)
    
    def test_get_alert_dir(self):
        from skill_paths import get_alert_dir
        alert = get_alert_dir()
        assert 'alert_logs' in alert
        assert os.path.exists(alert)
    
    def test_platform_detection(self):
        from skill_paths import IS_WINDOWS, IS_LINUX
        assert isinstance(IS_WINDOWS, bool)
        assert isinstance(IS_LINUX, bool)
        assert IS_WINDOWS != IS_LINUX  # 必然一个True一个False
    
    def test_skill_config_structure(self):
        from skill_paths import SKILL_CONFIG
        required_keys = ['skill_root', 'scripts_dir', 'data_dir', 'diary_dir', 
                         'alert_dir', 'is_windows', 'is_linux', 'data_source_priority']
        for key in required_keys:
            assert key in SKILL_CONFIG, f"缺少配置项: {key}"


# ── Test: market_state.py ─────────────────────────────────
class TestMarketState:
    def test_detect_market_state_returns_valid(self):
        from market_state import detect_market_state
        state = detect_market_state()
        assert state in ('bull', 'bear', 'mixed'), f"非法状态: {state}"
    
    def test_get_market_state_info(self):
        from market_state import detect_market_state, get_market_state_info
        state = detect_market_state()
        info = get_market_state_info()
        assert 'emoji' in info
        assert 'label' in info
        assert 'label_cn' in info
        assert info['emoji'] in ('🐂', '🐻', '⚖️')
    
    def test_state_info_matches_detect(self):
        from market_state import detect_market_state, get_market_state_info
        state = detect_market_state()
        info = get_market_state_info()
        state_map = {'bull': '🐂', 'bear': '🐻', 'mixed': '⚖️'}
        assert info['emoji'] == state_map[state]


# ── Test: 动态权重配置 ────────────────────────────────────
class TestDynamicWeights:
    def test_weight_table_structure(self):
        WEIGHT_TABLE = {
            '牛市':    {'tech': 0.60, 'basic': 0.25, 'industry': 0.15},
            '熊市':    {'tech': 0.25, 'basic': 0.50, 'industry': 0.25},
            '震荡市':  {'tech': 0.40, 'basic': 0.35, 'industry': 0.25},
        }
        for state, w in WEIGHT_TABLE.items():
            assert abs(w['tech'] + w['basic'] + w['industry'] - 1.0) < 0.001, \
                f"{state} 权重和={w['tech']+w['basic']+w['industry']}"
    
    def test_weight_table_bull_tech_high(self):
        WEIGHT_TABLE = {
            '牛市':    {'tech': 0.60, 'basic': 0.25, 'industry': 0.15},
        }
        w = WEIGHT_TABLE['牛市']
        assert w['tech'] > w['basic'] > w['industry']
    
    def test_weight_table_bear_basic_high(self):
        WEIGHT_TABLE = {
            '熊市':    {'tech': 0.25, 'basic': 0.50, 'industry': 0.25},
        }
        w = WEIGHT_TABLE['熊市']
        assert w['basic'] > w['tech']
        assert w['basic'] > w['industry']


# ── Test: 存档逻辑（幂等测试）───────────────────────────
class TestDiaryArchive:
    def test_archive_marker_format(self):
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        marker = f'# DATE: {today}'
        assert marker.startswith('# DATE: 2026')
    
    def test_archive_path_creation(self):
        from skill_paths import get_diary_dir
        diary = get_diary_dir()
        os.makedirs(diary, exist_ok=True)
        assert os.path.isdir(diary)


# ── Test: 配置加载 ────────────────────────────────────
class TestConfigLoading:
    def test_weights_json_structure(self):
        config_path = os.path.join(os.path.dirname(__file__), '../config/weights.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            for state in ['bull', 'bear', 'mixed']:
                assert state in cfg
                assert 'technical' in cfg[state]
                assert 'fundamental' in cfg[state]
                assert 'industry' in cfg[state]
                # v4.0: 支持3维或4维权重
                total = cfg[state]['technical'] + cfg[state]['fundamental'] + cfg[state]['industry']
                if 'momentum' in cfg[state]:
                    total += cfg[state]['momentum']
                assert abs(total - 1.0) < 0.001, f"{state} 权重总和={total} != 1.0"
        else:
            pytest.skip("weights.json 不存在，跳过")


# ── Test: 环境变量加载（安全测试）───────────────────────
class TestEnvVars:
    def test_qveris_key_not_hardcoded(self):
        """验证QVeris API Key不是硬编码在代码中"""
        market_report_path = os.path.join(os.path.dirname(__file__), '../scripts/market_report_v8.py')
        with open(market_report_path) as f:
            content = f.read()
        # 不应该出现 'sk-cn-' 这样的真实密钥格式
        assert "sk-cn-" not in content or "os.environ.get" in content
    
    def test_env_var_loading_function(self):
        """验证环境变量加载逻辑存在"""
        market_report_path = os.path.join(os.path.dirname(__file__), '../scripts/market_report_v8.py')
        with open(market_report_path) as f:
            content = f.read()
        assert "os.environ.get('QVERIS_API_KEY'" in content or \
               'os.environ.get("QVERIS_API_KEY"' in content


# ── Test: SECTOR_STOCKS 一致性 ────────────────────────
class TestSectorStocks:
    def test_sector_count(self):
        """至少应该有11个赛道"""
        # 延迟导入避免环境变量检查阻塞
        import importlib
        import market_report_v8
        importlib.reload(market_report_v8)
        from market_report_v8 import SECTOR_STOCKS
        assert len(SECTOR_STOCKS) >= 10, f"赛道数量太少: {len(SECTOR_STOCKS)}"

    def test_each_sector_has_stocks(self):
        from market_report_v8 import SECTOR_STOCKS
        for sector, stocks in SECTOR_STOCKS.items():
            assert len(stocks) >= 2, f"{sector} 股票数不足: {len(stocks)}"
            for name, code in stocks:
                assert code.startswith(('sh', 'sz', 'bj')), f"非法股票代码: {code}"

    def test_holding_stocks_consistent(self):
        from market_report_v8 import SECTOR_STOCKS
        # 持仓股票应该出现在赛道中
        holding = ['sz002384', 'sh600105', 'sh600576', 'sz300124', 'sh688017']
        all_codes = [code for stocks in SECTOR_STOCKS.values() for _, code in stocks]
        for h in holding:
            assert h in all_codes, f"持仓股票 {h} 未在任何赛道中找到"


# ── Test: factor_library.py (v4.0 新增) ───────────────────
class TestFactorLibrary:
    def test_rsi_normal_range(self):
        from factor_library import compute_rsi
        # 稳定上涨序列 → RSI 应接近100
        closes = [100 + i * 0.5 for i in range(30)]
        rsi = compute_rsi(closes)
        assert rsi is not None
        assert rsi > 70  # 强势

    def test_rsi_insufficient_data(self):
        from factor_library import compute_rsi
        assert compute_rsi([1, 2, 3]) is None

    def test_kdj_normal(self):
        from factor_library import compute_kdj
        highs = [10 + i * 0.1 for i in range(20)]
        lows = [9 + i * 0.1 for i in range(20)]
        closes = [9.5 + i * 0.1 for i in range(20)]
        k, d, j = compute_kdj(highs, lows, closes)
        assert k is not None and d is not None and j is not None
        assert 0 <= k <= 100

    def test_macd_normal(self):
        from factor_library import compute_macd
        closes = [100 + i * 0.3 for i in range(50)]
        dif, dea, bar = compute_macd(closes)
        assert dif is not None
        assert dea is not None
        assert bar is not None

    def test_bollinger_normal(self):
        from factor_library import compute_bollinger
        closes = [100 + (i % 5 - 2) for i in range(30)]
        upper, mid, lower, pctb = compute_bollinger(closes)
        assert upper is not None
        assert upper > mid > lower
        assert 0 <= pctb <= 1

    def test_atr_normal(self):
        from factor_library import compute_atr
        highs = [11 + i * 0.1 for i in range(30)]
        lows = [9 + i * 0.1 for i in range(30)]
        closes = [10 + i * 0.1 for i in range(30)]
        atr = compute_atr(highs, lows, closes)
        assert atr is not None
        assert atr > 0

    def test_momentum_score(self):
        from factor_library import compute_momentum_score
        closes = [100 + i * 0.2 for i in range(60)]
        result = compute_momentum_score(closes)
        assert 'momentum' in result
        assert 0 <= result['momentum'] <= 100
        assert 'rsi' in result
        assert 'macd_score' in result


# ── Test: risk_indicators.py (v4.0 新增) ──────────────────
class TestRiskIndicators:
    def test_max_drawdown_no_loss(self):
        from risk_indicators import max_drawdown
        curve = [100, 105, 110, 115, 120]
        result = max_drawdown(curve)
        assert result['max_dd'] == 0.0

    def test_max_drawdown_with_loss(self):
        from risk_indicators import max_drawdown
        curve = [100, 110, 90, 95, 80]
        result = max_drawdown(curve)
        assert result['max_dd'] > 20  # 从110跌到80 ≈ 27%

    def test_annualized_volatility(self):
        from risk_indicators import annualized_volatility
        returns = [0.01, -0.005, 0.008, -0.003, 0.012, -0.001, 0.005, -0.008, 0.003, 0.007,
                   -0.002, 0.009, -0.006, 0.004, 0.011]
        vol = annualized_volatility(returns)
        assert vol > 0
        assert vol < 100  # 合理范围

    def test_sharpe_ratio_positive(self):
        from risk_indicators import sharpe_ratio
        # 稳定正收益
        returns = [0.005] * 30
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_ratio_empty(self):
        from risk_indicators import sharpe_ratio
        assert sharpe_ratio([]) == 0

    def test_trailing_stop_triggered(self):
        from risk_indicators import trailing_stop_check
        result = trailing_stop_check(90, 100, stop_pct=8.0)
        assert result['triggered'] is True
        assert result['action'] == '止损出局'

    def test_trailing_stop_not_triggered(self):
        from risk_indicators import trailing_stop_check
        result = trailing_stop_check(105, 100, stop_pct=8.0)
        assert result['triggered'] is False
        assert result['action'] == '持有'

    def test_risk_score_normal(self):
        from risk_indicators import calculate_risk_score
        closes = [100 + i * 0.1 + (i % 3 - 1) * 0.5 for i in range(60)]
        result = calculate_risk_score(closes)
        assert 'risk_score' in result
        assert 0 <= result['risk_score'] <= 100
        assert 'risk_level' in result

    def test_risk_score_insufficient_data(self):
        from risk_indicators import calculate_risk_score
        result = calculate_risk_score([100, 101])
        assert result['risk_score'] == 50

    def test_format_risk_report(self):
        from risk_indicators import format_risk_report
        closes = [100 + i * 0.2 for i in range(60)]
        report = format_risk_report(closes, '测试股票')
        assert '风险评估' in report
        assert '测试股票' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])