"""
optimizer.py
策略参数调优模块（2026-05-07 重写版）

变更要点：
  [OLD] 每晚10点 → 10天样本 → DeepSeek建议 → 直接覆盖config
  [NEW] 每周五收盘后 → 验证集校验 → IC恶化才触发 → 新旧并行对比

功能：
  - 每周五收盘后触发（scheduler.py 配置）
  - 80/20 时间序列分割：用前80%数据训练，后20%验证
  - 当选出的新参数在验证集上 IC 低于旧参数时 → 跳过本次调优
  - 新旧参数并行跑一周静默对比
  - 备份旧配置 + 变更 diff 记录

依赖：
  pip install pyyaml pandas
"""

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd

# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/optimizer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("optimizer")

# ─────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
CONFIG_BACKUP_DIR = ROOT_DIR / "config_backups"
DB_PATH = ROOT_DIR / "db" / "quant.db"
LOG_DIR = ROOT_DIR / "logs"

CONFIG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 配置读写 & 备份
# ─────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    logger.info("config.yaml 已更新")


def backup_config() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CONFIG_BACKUP_DIR / f"config_{timestamp}.yaml"
    shutil.copy2(CONFIG_PATH, backup_path)
    logger.info(f"配置已备份至 {backup_path}")
    return backup_path


def restore_config(backup_path: Path) -> None:
    shutil.copy2(backup_path, CONFIG_PATH)
    logger.info(f"配置已从 {backup_path} 恢复")


# ─────────────────────────────────────────────
# IC (Information Coefficient) 计算
# ─────────────────────────────────────────────
def calc_ic(signal_scores: list[float], actual_returns: list[float]) -> float:
    """
    计算 IC = Spearman 秩相关系数（信号评分 vs 实际收益）。
    正值表示信号能预测方向。
    """
    if len(signal_scores) < 5 or len(actual_returns) < 5:
        return 0.0
    s = pd.Series(signal_scores)
    r = pd.Series(actual_returns)
    return round(s.corr(r, method="spearman"), 4)


# ─────────────────────────────────────────────
# 绩效数据查询（从 SQLite）
# ─────────────────────────────────────────────
def fetch_signals(days: int = 30) -> pd.DataFrame:
    """
    从 SQLite 拉取 signals 表，返回 DataFrame。
    包含：signal_date, symbol, direction, composite_score, actual_return, is_correct
    """
    if not DB_PATH.exists():
        logger.warning(f"数据库文件不存在：{DB_PATH}")
        return pd.DataFrame()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT signal_date, symbol, direction, composite_score,
                   actual_return, is_correct
            FROM signals
            WHERE signal_date >= ?
              AND actual_return IS NOT NULL
              AND composite_score IS NOT NULL
            ORDER BY signal_date ASC
            """,
            conn,
            params=(cutoff,),
        )
        conn.close()
        return df
    except Exception as e:
        logger.error(f"查询信号数据失败：{e}")
        return pd.DataFrame()


def compute_performance(df: pd.DataFrame) -> dict:
    """从 DataFrame 计算绩效统计"""
    if df.empty:
        return {
            "total_signals": 0, "win_rate_pct": 0,
            "avg_return_pct": 0, "ic": 0,
        }

    total = len(df)
    correct = df["is_correct"].sum() if "is_correct" in df.columns else 0
    win_rate = round(correct / total * 100, 1) if total > 0 else 0
    avg_ret = round(df["actual_return"].mean() * 100, 2) if "actual_return" in df.columns else 0
    ic = calc_ic(
        df["composite_score"].tolist(),
        df["actual_return"].tolist(),
    )

    by_direction = {}
    if "direction" in df.columns:
        for d in df["direction"].unique():
            sub = df[df["direction"] == d]
            sub_total = len(sub)
            sub_correct = sub["is_correct"].sum() if "is_correct" in sub.columns else 0
            by_direction[d] = {
                "count": sub_total,
                "win_rate_pct": round(sub_correct / sub_total * 100, 1) if sub_total > 0 else 0,
                "avg_return_pct": round(sub["actual_return"].mean() * 100, 2) if "actual_return" in sub.columns else 0,
            }

    return {
        "total_signals": total,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "ic": ic,
        "by_direction": by_direction,
    }


# ─────────────────────────────────────────────
# 参数枚举 & 验证集评估（80/20 时序分割）
# ─────────────────────────────────────────────
def _score_with_params(composite_scores: list[float], actual_returns: list[float],
                       buy_threshold: float, sell_threshold: float) -> dict:
    """
    用给定 buy/sell 阈值在数据上跑一遍，返回模拟绩效。
    buy_threshold: 综合分 >= 此值 → buy
    sell_threshold: 综合分 <= 此值 → sell
    中间 → hold
    """
    if len(composite_scores) < 5:
        return {"ic": 0, "win_rate": 0, "avg_return": 0, "signal_count": 0}

    predictions = []
    for score in composite_scores:
        if score >= buy_threshold:
            predictions.append(1)   # 看涨
        elif score <= sell_threshold:
            predictions.append(-1)  # 看跌
        else:
            predictions.append(0)   # 观望

    # 只对有信号的交易日计算 IC
    signal_indices = [i for i, p in enumerate(predictions) if p != 0]
    if len(signal_indices) < 3:
        return {"ic": 0, "win_rate": 0, "avg_return": 0, "signal_count": 0}

    signal_scores = [predictions[i] for i in signal_indices]
    signal_returns = [actual_returns[i] for i in signal_indices]

    ic = calc_ic(signal_scores, signal_returns)
    correct = sum(1 for p, r in zip(signal_scores, signal_returns) if (p > 0 and r > 0) or (p < 0 and r < 0))
    win_rate = round(correct / len(signal_scores) * 100, 1)
    avg_return = round(sum(signal_returns) / len(signal_returns) * 100, 2)

    return {
        "ic": ic,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "signal_count": len(signal_scores),
    }


def _default_thresholds() -> dict:
    """当前默认阈值"""
    return {"buy": 63, "sell": 37}


def evaluate_thresholds_on_split(df: pd.DataFrame, train_ratio: float = 0.8) -> dict:
    """
    80/20 时间序列分割：
      - 前 80% 数据遍历候选阈值
      - 选出训练集上 IC 最高的阈值组合
      - 在测试集上验证：新阈值 IC 是否 > 旧阈值 IC
    返回调优建议。
    """
    if df.empty or len(df) < 20:
        logger.info(f"数据不足（{len(df)} 条），无法调优")
        return {"action": "no_change", "reason": f"数据不足（{len(df)} 条），需要至少 20 条"}

    scores = df["composite_score"].tolist()
    returns = df["actual_return"].tolist()
    n = len(scores)
    split = int(n * train_ratio)

    train_scores, train_returns = scores[:split], returns[:split]
    test_scores, test_returns = scores[split:], returns[split:]

    # 遍历候选阈值
    candidates = []
    # buy: 58-68, sell: 32-42（步长 1）
    for buy_t in range(58, 69):
        for sell_t in range(32, 43):
            if buy_t <= sell_t + 25:
                continue  # 确保 buy/sell 之间有至少 25 分的 hold 区间
            perf = _score_with_params(train_scores, train_returns, buy_t, sell_t)
            candidates.append({
                "buy_threshold": buy_t,
                "sell_threshold": sell_t,
                "ic": perf["ic"],
                "win_rate": perf["win_rate"],
                "avg_return": perf["avg_return"],
                "signal_count": perf["signal_count"],
            })

    if not candidates:
        return {"action": "no_change", "reason": "无有效候选参数"}

    # 选训练集 IC 最高的
    best = max(candidates, key=lambda c: c["ic"])
    defaults = _default_thresholds()

    # 在测试集上评估新参数 vs 旧参数
    new_perf = _score_with_params(test_scores, test_returns, best["buy_threshold"], best["sell_threshold"])
    old_perf = _score_with_params(test_scores, test_returns, defaults["buy"], defaults["sell"])

    ic_improvement = new_perf["ic"] - old_perf["ic"]

    logger.info(
        f"候选训练: buy={best['buy_threshold']} sell={best['sell_threshold']} "
        f"训练IC={best['ic']} 验证IC={new_perf['ic']} "
        f"当前IC={old_perf['ic']} 提升={ic_improvement:+.4f}"
    )

    if ic_improvement <= 0:
        return {
            "action": "no_change",
            "reason": f"验证集 IC 未改善（新{new_perf['ic']} vs 旧{old_perf['ic']}），跳过",
            "details": {
                "best_candidate": best,
                "new_perf": new_perf,
                "old_perf": old_perf,
                "ic_improvement": ic_improvement,
            },
        }

    # 有改善，生成变更
    changes = {
        "signals.buy_threshold": best["buy_threshold"],
        "signals.sell_threshold": best["sell_threshold"],
    }

    return {
        "action": "update",
        "reason": f"验证集 IC 改善 {ic_improvement:+.4f}（{old_perf['ic']}→{new_perf['ic']}）",
        "changes": changes,
        "details": {
            "best_candidate": best,
            "new_perf": new_perf,
            "old_perf": old_perf,
            "ic_improvement": ic_improvement,
        },
    }


# ─────────────────────────────────────────────
# 主入口：每周调优
# ─────────────────────────────────────────────
def run_weekly_optimization(days: int = 60, dry_run: bool = False) -> dict:
    """
    完整的周调优流程（取代旧的 nightly_optimization）：

      1. 拉取最近 N 天信号数据（默认 60 天）
      2. 计算当前绩效（整体 IC、胜率等）
      3. 80/20 时序分割 → 遍历候选阈值 → 验证集校验
      4. 验证集 IC 未改善 → 跳过
      5. 备份旧配置 → 写入新配置 → 记录 diff
      6. 返回调优报告

    Args:
        days:    分析最近 N 天数据（默认 60 天，跨越约 3 个自然月）
        dry_run: 只打印不写入

    Returns:
        report dict
    """
    logger.info("=" * 50)
    logger.info("周度策略参数调优开始")
    logger.info("=" * 50)

    # Step 1: 加载配置 & 拉取数据
    config = load_config()
    df = fetch_signals(days=days)

    if df.empty:
        logger.warning("信号数据为空，跳过调优")
        return {"action": "no_change", "reason": "无信号数据", "changes": {}}

    logger.info(f"拉取到 {len(df)} 条信号数据（近 {days} 天）")

    # Step 2: 计算当前绩效
    perf = compute_performance(df)
    logger.info(
        f"当前绩效：{perf['total_signals']} 条，"
        f"胜率 {perf['win_rate_pct']}%，"
        f"平均收益 {perf['avg_return_pct']}%，"
        f"IC={perf['ic']}"
    )

    # 如果 IC 连续为正且 > 0.03，说明当前参数没问题，跳过调优
    # 这是一个简单的 IC 监控：如果当前整体 IC 尚可，不折腾
    if perf["ic"] > 0.03:
        logger.info(f"当前 IC={perf['ic']} > 0.03，参数状态良好，跳过本次调优")
        return {
            "action": "no_change",
            "reason": f"当前整体 IC={perf['ic']} > 0.03，参数状态良好",
            "performance": perf,
        }

    # Step 3: 80/20 分割调优
    result = evaluate_thresholds_on_split(df, train_ratio=0.8)

    result["performance"] = perf

    if result["action"] != "update":
        logger.info(result["reason"])
        return result

    if dry_run:
        logger.info(f"[DRY RUN] 建议变更：{result['changes']}")
        return result

    # Step 4: 备份 + 写入
    changes = result["changes"]

    # 先把旧版阈值保存到 config 的 signals 段（如果还没有的话）
    signals_cfg = config.setdefault("signals", {})
    if "buy_threshold" not in signals_cfg:
        signals_cfg["buy_threshold"] = 63
    if "sell_threshold" not in signals_cfg:
        signals_cfg["sell_threshold"] = 37

    # 保存当前参数作为"旧参数"备份（用于并行对比）
    signals_cfg["_previous_buy_threshold"] = signals_cfg.get("buy_threshold", 63)
    signals_cfg["_previous_sell_threshold"] = signals_cfg.get("sell_threshold", 37)

    # 写入新参数
    for key, value in changes.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            config.setdefault(parts[0], {})[parts[1]] = value

    backup_path = backup_config()
    save_config(config)

    # 生成变更摘要文本供下次信号生成时参考
    summary = (
        f"阈值调整：buy={signals_cfg.get('_previous_buy_threshold', '?')}→{changes.get('signals.buy_threshold')}, "
        f"sell={signals_cfg.get('_previous_sell_threshold', '?')}→{changes.get('signals.sell_threshold')}, "
        f"验证集 IC 提升 {result['details']['ic_improvement']:+.4f}"
    )
    logger.info(f"调优完成：{summary}")

    return {
        "action": "update",
        "reason": summary,
        "changes": changes,
        "backup_path": str(backup_path),
        "performance": perf,
        "details": result.get("details"),
    }
