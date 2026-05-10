"""
strategy/ic_monitor.py
IC（信息系数）、ICIR、Spearman Rank IC 每日监控

功能：
  - 每日从 signals 表提取近 N 天的 composite_score 和 actual_return
  - 计算 Pearson IC（线性）、Spearman Rank IC（排序）、ICIR（IC均值/标准差）
  - benchmark_return 不为空时计算 excess_return（超额收益）
  - 超过阈值时写入监控JSON报警（供 monitor_daemon 推送到微信）

阈值（config.yaml ic_monitor 段）：
  - ic_alert_threshold: 0.25    # |IC| > 0.25 报警（过拟合或异常）
  - icir_alert_threshold: 2.5   # |ICIR| > 2.5 报警（不稳定）
  - ic_warn_threshold: 0.15     # |IC| > 0.15 warning日志
  - min_signals: 10             # 最少信号条数，低于此跳过

用法：
  python strategy/ic_monitor.py                          # 默认近30天
  python strategy/ic_monitor.py --days 60 --alert-file /tmp/ic_alert.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from storage.db import DB_PATH

logger = logging.getLogger("ic_monitor")

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
MONITOR_DIR = ROOT_DIR / "monitor"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "ic_monitor": {
        "ic_alert_threshold": 0.25,
        "icir_alert_threshold": 2.5,
        "ic_warn_threshold": 0.15,
        "min_signals": 10,
        "days": 30,
    }
}


def load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("ic_monitor", DEFAULT_CONFIG["ic_monitor"])
    except Exception as e:
        logger.warning(f"读取config失败: {e}")
    return DEFAULT_CONFIG["ic_monitor"]


def fetch_signals_data(days: int = 30) -> pd.DataFrame:
    """从 signals 表提取近 N 天的评分和收益数据"""
    if not DB_PATH.exists():
        logger.warning("数据库不存在")
        return pd.DataFrame()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT symbol, signal_date, composite_score, actual_return, benchmark_return
            FROM signals
            WHERE signal_date >= ?
              AND composite_score IS NOT NULL
              AND actual_return IS NOT NULL
            ORDER BY signal_date
            """,
            conn, params=(cutoff,),
        )
        conn.close()

        if df.empty:
            logger.info(f"近 {days} 天无完整信号数据（含评分的信号）")
            return df

        logger.info(f"从 signals 表读取 {len(df)} 条信号记录（含评分+收益）")
        return df
    except Exception as e:
        logger.error(f"查询信号数据失败: {e}")
        return pd.DataFrame()


def compute_ic_metrics(df: pd.DataFrame, config: dict) -> dict:
    """
    计算多个IC指标。

    Returns:
        {
            "pearson_ic": float,        # 全量 Pearson IC
            "spearman_ic": float,       # 全量 Spearman Rank IC
            "icir": float,              # Pearson ICIR（多日IC均值/标准差）
            "n": int,                   # 信号总数
            "by_symbol": {...},         # 每个标的独立IC
            "excess_pearson_ic": float or None,  # 超额收益IC（扣除沪深300）
            "warnings": [str],
            "alerts": [str],
        }
    """
    cfg = config if isinstance(config, dict) else {}
    ic_alert = cfg.get("ic_alert_threshold", 0.25)
    icir_alert = cfg.get("icir_alert_threshold", 2.5)
    ic_warn = cfg.get("ic_warn_threshold", 0.15)

    result = {
        "compute_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pearson_ic": None,
        "spearman_ic": None,
        "icir": None,
        "n": len(df),
        "by_symbol": {},
        "excess_pearson_ic": None,
        "warnings": [],
        "alerts": [],
    }

    n = len(df)
    if n < cfg.get("min_signals", 10):
        result["warnings"].append(f"信号不足 {n} 条（需≥{cfg.get('min_signals', 10)}），跳过IC计算")
        return result

    scores = df["composite_score"].values.astype(float)
    returns = df["actual_return"].values.astype(float)

    # ── 1. Pearson IC ────────────────────────
    try:
        p_ic, p_pval = pearsonr(scores, returns)
        result["pearson_ic"] = round(float(p_ic), 4)
        result["pearson_p_value"] = round(float(p_pval), 6)
    except Exception as e:
        logger.warning(f"Pearson IC 计算失败: {e}")
        result["pearson_ic"] = 0.0
        result["pearson_p_value"] = 1.0

    # ── 2. Spearman Rank IC ─────────────────
    try:
        s_ic, s_pval = spearmanr(scores, returns)
        result["spearman_ic"] = round(float(s_ic), 4)
        result["spearman_p_value"] = round(float(s_pval), 6)
    except Exception as e:
        logger.warning(f"Spearman IC 计算失败: {e}")
        result["spearman_ic"] = 0.0
        result["spearman_p_value"] = 1.0

    # ── 3. 分日IC → ICIR ────────────────────
    try:
        df_by_date = df.copy()
        df_by_date["date"] = pd.to_datetime(df_by_date["signal_date"])
        daily_ic = df_by_date.groupby("date").apply(
            lambda g: pearsonr(g["composite_score"].astype(float),
                               g["actual_return"].astype(float))[0],
            include_groups=False,
        )
        if len(daily_ic) > 0:
            ic_mean = daily_ic.mean()
            ic_std = daily_ic.std()
            result["icir"] = round(float(ic_mean / ic_std), 4) if ic_std > 0 else 0.0
            result["daily_ic_count"] = len(daily_ic)
            result["daily_ic_mean"] = round(float(ic_mean), 4)
            result["daily_ic_std"] = round(float(ic_std), 4)
        else:
            result["icir"] = 0.0
            result["daily_ic_count"] = 0
    except Exception as e:
        logger.warning(f"ICIR 计算失败: {e}")
        result["icir"] = 0.0

    # ── 4. 超额收益 IC（相对沪深300）─────────
    if "benchmark_return" in df.columns and df["benchmark_return"].notna().any():
        try:
            excess = df["actual_return"].astype(float) - df["benchmark_return"].astype(float)
            exc_ic, _ = pearsonr(scores, excess)
            result["excess_pearson_ic"] = round(float(exc_ic), 4)
        except Exception:
            result["excess_pearson_ic"] = None

    # ── 5. 分标的IC ──────────────────────────
    try:
        for symbol, grp in df.groupby("symbol"):
            if len(grp) < 5:
                continue
            try:
                sym_ic, _ = pearsonr(grp["composite_score"].astype(float),
                                     grp["actual_return"].astype(float))
                result["by_symbol"][symbol] = round(float(sym_ic), 4)
            except Exception:
                continue
    except Exception:
        pass

    # ── 6. 阈值判断 ──────────────────────────
    ic_val = result["pearson_ic"]
    if ic_val is not None and abs(ic_val) > ic_alert:
        result["alerts"].append(
            f"⚠️ |Pearson IC|={abs(ic_val):.4f} 超过报警阈值 {ic_alert} "
            f"({'正向过拟合' if ic_val > 0 else '反向异常'})"
        )
    elif ic_val is not None and abs(ic_val) > ic_warn:
        result["warnings"].append(
            f"⚡ |Pearson IC|={abs(ic_val):.4f} 超过关注阈值 {ic_warn}"
        )

    icir_val = result["icir"]
    if icir_val is not None and abs(icir_val) > icir_alert:
        result["alerts"].append(
            f"⚠️ |ICIR|={abs(icir_val):.4f} 超过报警阈值 {icir_alert} "
            f"（IC不稳定：日均IC波动大）"
        )

    return result


def write_alert_json(result: dict, alert_file: Optional[str] = None) -> Optional[str]:
    """有alerts时写入监控报警JSON文件"""
    if not result.get("alerts"):
        return None
    if not alert_file:
        alert_file = str(MONITOR_DIR / "ic_alert.json")

    alert_data = {
        "type": "ic_monitor",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alerts": result["alerts"],
        "warnings": result.get("warnings", []),
        "pearson_ic": result.get("pearson_ic"),
        "spearman_ic": result.get("spearman_ic"),
        "icir": result.get("icir"),
        "n_signals": result.get("n", 0),
    }

    with open(alert_file, "w", encoding="utf-8") as f:
        json.dump(alert_data, f, ensure_ascii=False, indent=2)

    logger.info(f"IC报警已写入 {alert_file}")
    return alert_file


def run_ic_monitor(days: int = 30, alert_file: Optional[str] = None, dry_run: bool = False) -> dict:
    """
    主入口：跑IC/ICIR监控。

    Returns:
        完整IC指标Dict
    """
    config = load_config()
    if not dry_run:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(str(ROOT_DIR / "logs" / "ic_monitor.log"), encoding="utf-8"),
            ],
            force=True,
        )

    logger.info("=" * 50)
    logger.info(f"IC监控开始（近 {days} 天）")
    logger.info("=" * 50)

    df = fetch_signals_data(days=days)
    if df.empty:
        logger.info("无信号数据，跳过IC监控")
        return {"status": "skipped", "reason": "无信号数据"}

    result = compute_ic_metrics(df, config)

    # 日志输出
    if result["pearson_ic"] is not None:
        logger.info(
            f"Pearson IC={result['pearson_ic']}  "
            f"p={result.get('pearson_p_value', 'N/A')}  "
            f"Spearman IC={result.get('spearman_ic', 'N/A')}"
        )
    if result["icir"] is not None:
        logger.info(
            f"ICIR={result['icir']}  "
            f"（{result.get('daily_ic_count', 0)} 日均IC）"
        )
    if result["excess_pearson_ic"] is not None:
        logger.info(f"超额收益IC（vs沪深300）={result['excess_pearson_ic']}")
    if result["alerts"]:
        logger.warning(f"🚨 {len(result['alerts'])} 条IC报警!")
        for a in result["alerts"]:
            logger.warning(f"  {a}")
        if not dry_run:
            write_alert_json(result, alert_file)
    elif result["warnings"]:
        logger.warning(f"⚠️ {len(result['warnings'])} 条IC关注提示")
        for w in result["warnings"]:
            logger.warning(f"  {w}")
    else:
        logger.info("✅ IC指标在正常范围内")

    logger.info("IC监控完成")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IC/ICIR 每日监控")
    parser.add_argument("--days", type=int, default=30, help="回溯天数")
    parser.add_argument("--alert-file", type=str, default=None, help="报警JSON路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印不写文件")
    args = parser.parse_args()
    run_ic_monitor(days=args.days, alert_file=args.alert_file, dry_run=args.dry_run)
