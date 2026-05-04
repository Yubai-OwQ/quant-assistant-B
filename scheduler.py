"""
scheduler.py - 主调度器
APScheduler 驱动的 24h 自动运行循环
同时启动 Gradio 对话界面
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import (
    fetch_stock_data, fetch_fear_greed_index, get_latest_northbound, update_all
)
from strategy.signals import calculate_all_signals, run_backtest, run_all
from strategy.ai_engine import analyze_market, nightly_optimize
from strategy.optimizer import run_nightly_optimization
from storage.db import init_db, save_signal, save_performance, save_config_version, get_performance_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quant_assistant.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# 全局状态（线程安全读取，Gradio界面共享）
_state = {
    "config": {},
    "latest_signals": {},
    "latest_analysis": {},
    "last_run": None,
    "status": "初始化中"
}
_state_lock = threading.Lock()


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, reason: str = "调度器更新"):
    config["last_updated"] = time.strftime("%Y-%m-%d")
    config["version"] = config.get("version", 1) + 1
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    save_config_version(config, reason)
    logger.info(f"配置已保存（版本 {config['version']}）")


def run_daily_analysis():
    """每日主分析任务（每天09:30触发）"""
    logger.info("=== 开始每日AI分析 ===")
    with _state_lock:
        config = _state["config"]
        _state["status"] = "分析中"

    symbols = config.get("strategy", {}).get("symbols", ["510300"])
    market = config.get("strategy", {}).get("market", "A股")

    # Step 1: 拉取最新行情 & 北向资金 & 市场情绪
    update_all()
    fetch_fear_greed_index(config)

    all_signals = {}
    all_analyses = {}

    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, market=market, days=120)
            signals = calculate_all_signals(df, config)
            northbound = get_latest_northbound()
            fear_greed = fetch_fear_greed_index(config) or {}
            analysis = analyze_market(signals, fear_greed, config, northbound=northbound)
            perf = run_backtest(df, config, window=30)

            save_signal(symbol, signals, analysis)
            save_performance(symbol, perf, config.get("version", 1))

            all_signals[symbol] = signals
            all_analyses[symbol] = analysis

            logger.info(
                f"[{symbol}] 方向={analysis.get('direction')} "
                f"置信度={analysis.get('confidence', 0):.0%} "
                f"建议仓位={analysis.get('suggested_position', 0):.0%} "
                f"Sharpe={perf.get('sharpe', 0):.2f}"
            )
        except Exception as e:
            logger.error(f"分析 {symbol} 失败: {e}")

    # Step 2: 统一生成今日信号（写入 signals 表供 UI 查询）
    try:
        run_all(dry_run=False)
    except Exception as e:
        logger.error(f"信号生成失败: {e}")

    with _state_lock:
        _state["latest_signals"] = all_signals
        _state["latest_analysis"] = all_analyses
        _state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _state["status"] = "就绪"

    logger.info("=== 每日分析完成 ===")


def run_nightly_optimize():
    """夜间参数调优任务（每天22:00触发）"""
    logger.info("=== 开始夜间AI调优 ===")
    with _state_lock:
        config = _state["config"].copy()

    # 使用 optimizer.py 的全功能调优（含备份、校验、写入）
    report = run_nightly_optimization(days=10, dry_run=False)

    if report.get("action") == "update":
        with _state_lock:
            try:
                import yaml
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    _state["config"] = yaml.safe_load(f)
            except Exception:
                pass
        logger.info(f"调优完成，变更: {report.get('changes', {})}")
        logger.info(f"AI评估: {report.get('reason', '')}")
    else:
        logger.info(f"AI判断无需调整: {report.get('reason', '保持现状')}")

    logger.info("=== 夜间调优完成 ===")


def get_state():
    """线程安全地获取全局状态"""
    with _state_lock:
        return dict(_state)


def update_config_from_chat(new_config: dict):
    """从对话界面更新配置"""
    save_config(new_config, reason="对话界面更新")
    with _state_lock:
        _state["config"] = new_config


def start_scheduler():
    """启动APScheduler后台调度器"""
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
    cfg = _state["config"].get("scheduler", {})

    fetch_time = cfg.get("analysis_time", "09:30")
    h, m = fetch_time.split(":")
    scheduler.add_job(run_daily_analysis, CronTrigger(hour=int(h), minute=int(m)),
                      id="daily_analysis", name="每日AI分析")

    opt_time = cfg.get("optimize_time", "22:00")
    h2, m2 = opt_time.split(":")
    scheduler.add_job(run_nightly_optimize, CronTrigger(hour=int(h2), minute=int(m2)),
                      id="nightly_optimize", name="夜间AI调优")

    scheduler.start()
    logger.info(f"调度器已启动 → 每日分析: {fetch_time}，夜间调优: {opt_time}")
    return scheduler


if __name__ == "__main__":
    logger.info("量化助手启动中...")

    init_db()
    config = load_config()

    with _state_lock:
        _state["config"] = config
        _state["status"] = "就绪"

    logger.info("执行启动时首次分析...")
    run_daily_analysis()

    scheduler = start_scheduler()

    logger.info("启动 Gradio 界面...")
    try:
        from app import create_gradio_app
        app = create_gradio_app()
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except KeyboardInterrupt:
        logger.info("收到退出信号，停止调度器")
        scheduler.shutdown()
