"""
optimizer.py
夜间策略参数自动调优模块

功能：
  - 读取近期信号绩效数据
  - 将绩效摘要发送给 DeepSeek AI
  - 解析 AI 返回的参数调优建议
  - 版本化备份旧参数，写入新参数到 config.yaml
  - 支持手动触发和 APScheduler 定时触发

依赖：
  pip install openai pyyaml
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
from openai import OpenAI

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
# 路径常量（项目根目录相对路径）
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
CONFIG_BACKUP_DIR = ROOT_DIR / "config_backups"
DB_PATH = ROOT_DIR / "db" / "quant.db"
LOG_DIR = ROOT_DIR / "logs"

# 确保必要目录存在
CONFIG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# DeepSeek 客户端初始化
# ─────────────────────────────────────────────
def get_deepseek_client() -> OpenAI:
    """初始化 DeepSeek API 客户端（兼容 OpenAI SDK）"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


# ─────────────────────────────────────────────
# 配置文件读写
# ─────────────────────────────────────────────
def load_config() -> dict:
    """加载 config.yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    """写入 config.yaml（覆盖）"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    logger.info("config.yaml 已更新")


def backup_config() -> Path:
    """
    将当前 config.yaml 备份到 config_backups/config_YYYYMMDD_HHMMSS.yaml
    返回备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CONFIG_BACKUP_DIR / f"config_{timestamp}.yaml"
    shutil.copy2(CONFIG_PATH, backup_path)
    logger.info(f"配置已备份至 {backup_path}")
    return backup_path


def restore_config(backup_path: Path) -> None:
    """从指定备份文件恢复 config.yaml"""
    shutil.copy2(backup_path, CONFIG_PATH)
    logger.info(f"配置已从 {backup_path} 恢复")


def list_config_backups() -> list[Path]:
    """列出所有备份，按时间倒序"""
    backups = sorted(CONFIG_BACKUP_DIR.glob("config_*.yaml"), reverse=True)
    return backups


# ─────────────────────────────────────────────
# 绩效数据查询（从 SQLite）
# ─────────────────────────────────────────────
def fetch_recent_performance(days: int = 10) -> dict:
    """
    从 SQLite 查询最近 N 天的信号绩效统计。

    期望 signals 表结构：
      id, symbol, signal_date, direction, confidence,
      suggested_position, stop_loss, actual_return, is_correct

    返回 dict，包含：
      total_signals, win_rate, avg_return, avg_confidence,
      max_drawdown, by_direction, by_symbol
    """
    if not DB_PATH.exists():
        logger.warning(f"数据库文件不存在：{DB_PATH}，返回空绩效")
        return _empty_performance()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # ── 全局统计 ──────────────────────────────
        cur.execute(
            """
            SELECT
                COUNT(*)                                  AS total,
                AVG(CASE WHEN is_correct=1 THEN 1.0 ELSE 0.0 END) AS win_rate,
                AVG(actual_return)                        AS avg_return,
                AVG(confidence)                           AS avg_confidence,
                MIN(actual_return)                        AS worst_return
            FROM signals
            WHERE signal_date >= ?
              AND actual_return IS NOT NULL
            """,
            (cutoff,),
        )
        row = cur.fetchone()

        total = row["total"] or 0
        win_rate = round((row["win_rate"] or 0) * 100, 1)
        avg_return = round((row["avg_return"] or 0) * 100, 2)
        avg_confidence = round(row["avg_confidence"] or 0, 1)
        worst_return = round((row["worst_return"] or 0) * 100, 2)

        # ── 按方向拆分 ───────────────────────────
        cur.execute(
            """
            SELECT direction,
                   COUNT(*) AS cnt,
                   AVG(CASE WHEN is_correct=1 THEN 1.0 ELSE 0.0 END) AS wr,
                   AVG(actual_return) AS avg_ret
            FROM signals
            WHERE signal_date >= ? AND actual_return IS NOT NULL
            GROUP BY direction
            """,
            (cutoff,),
        )
        by_direction = {
            r["direction"]: {
                "count": r["cnt"],
                "win_rate": round((r["wr"] or 0) * 100, 1),
                "avg_return_pct": round((r["avg_ret"] or 0) * 100, 2),
            }
            for r in cur.fetchall()
        }

        # ── 按标的拆分 ───────────────────────────
        cur.execute(
            """
            SELECT symbol,
                   COUNT(*) AS cnt,
                   AVG(CASE WHEN is_correct=1 THEN 1.0 ELSE 0.0 END) AS wr,
                   AVG(actual_return) AS avg_ret
            FROM signals
            WHERE signal_date >= ? AND actual_return IS NOT NULL
            GROUP BY symbol
            ORDER BY avg_ret DESC
            """,
            (cutoff,),
        )
        by_symbol = {
            r["symbol"]: {
                "count": r["cnt"],
                "win_rate": round((r["wr"] or 0) * 100, 1),
                "avg_return_pct": round((r["avg_ret"] or 0) * 100, 2),
            }
            for r in cur.fetchall()
        }

        conn.close()

        return {
            "period_days": days,
            "total_signals": total,
            "win_rate_pct": win_rate,
            "avg_return_pct": avg_return,
            "avg_confidence": avg_confidence,
            "worst_single_return_pct": worst_return,
            "by_direction": by_direction,
            "by_symbol": by_symbol,
        }

    except sqlite3.Error as e:
        logger.error(f"查询绩效数据失败：{e}")
        return _empty_performance()


def _empty_performance() -> dict:
    return {
        "period_days": 0,
        "total_signals": 0,
        "win_rate_pct": 0,
        "avg_return_pct": 0,
        "avg_confidence": 0,
        "worst_single_return_pct": 0,
        "by_direction": {},
        "by_symbol": {},
    }


# ─────────────────────────────────────────────
# Prompt 构建
# ─────────────────────────────────────────────
STYLE_DESCRIPTIONS = {
    "aggressive": "激进——追求高收益，接受较大波动，仓位可达 80%，止损可宽至 8%",
    "balanced":   "均衡——收益与风险平衡，仓位 40–60%，止损 4–6%",
    "conservative": "保守——优先保本，仓位不超过 30%，止损不超过 3%",
}

# 各风格下参数的合法范围（用于校验 AI 输出）
STYLE_BOUNDS = {
    "aggressive":   {"max_position": (0.5, 1.0), "stop_loss": (0.03, 0.10), "take_profit": (0.08, 0.25), "rsi_overbought": (65, 80), "rsi_oversold": (20, 40)},
    "balanced":     {"max_position": (0.3, 0.7), "stop_loss": (0.02, 0.07), "take_profit": (0.06, 0.18), "rsi_overbought": (65, 78), "rsi_oversold": (22, 38)},
    "conservative": {"max_position": (0.1, 0.4), "stop_loss": (0.01, 0.04), "take_profit": (0.04, 0.12), "rsi_overbought": (68, 80), "rsi_oversold": (22, 35)},
}


def build_optimizer_prompt(config: dict, performance: dict) -> str:
    style = config.get("strategy", {}).get("style", "balanced")
    style_desc = STYLE_DESCRIPTIONS.get(style, style)
    bounds = STYLE_BOUNDS.get(style, STYLE_BOUNDS["balanced"])

    perf_json = json.dumps(performance, ensure_ascii=False, indent=2)
    current_signals = json.dumps(config.get("signals", {}), ensure_ascii=False, indent=2)
    current_risk = json.dumps(config.get("risk", {}), ensure_ascii=False, indent=2)

    return f"""你是专业的量化策略优化师。请根据近期绩效表现，对策略参数提出精准的调整建议。

## 当前策略风格
{style_desc}

## 当前参数
风险参数：
{current_risk}

信号参数：
{current_signals}

## 近 {performance['period_days']} 天绩效数据
{perf_json}

## 参数合法范围（必须严格遵守，不得超出）
{json.dumps(bounds, ensure_ascii=False, indent=2)}

## 调优原则
1. 如果胜率 < 45%，优先收紧止损和降低仓位上限
2. 如果平均收益 < 0%，检查 RSI 超买超卖阈值是否需要调整
3. 如果胜率 > 65% 且平均收益 > 0%，可适当放宽仓位上限以扩大盈利
4. 每次参数变动幅度不超过当前值的 20%，避免参数跳变
5. 只调整确实需要调整的参数，不需要变动的保持原值
6. 如果信号数量 < 5 条，数据不足，返回 no_change

## 输出格式（只输出 JSON，不要任何其他文字）
{{
  "action": "update" 或 "no_change",
  "reason": "50字以内的简短说明",
  "changes": {{
    "risk.max_position": <float 或 null>,
    "risk.stop_loss": <float 或 null>,
    "risk.take_profit": <float 或 null>,
    "signals.rsi_overbought": <int 或 null>,
    "signals.rsi_oversold": <int 或 null>,
    "signals.macd_weight": <float 或 null>,
    "signals.rsi_weight": <float 或 null>,
    "signals.volume_weight": <float 或 null>
  }}
}}
changes 中值为 null 表示该参数不做变动。
"""


# ─────────────────────────────────────────────
# 调用 DeepSeek
# ─────────────────────────────────────────────
def call_deepseek(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """
    调用 DeepSeek API，返回解析后的 JSON dict。
    失败时重试最多 max_retries 次，全部失败返回 None。
    """
    client = get_deepseek_client()

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"调用 DeepSeek（第 {attempt} 次）...")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,    # 参数调优要稳定，不需要创造性
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            logger.debug(f"DeepSeek 原始返回：{raw}")
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败（第 {attempt} 次）：{e}，原文：{raw[:200]}")
        except Exception as e:
            logger.warning(f"API 调用异常（第 {attempt} 次）：{e}")

    logger.error("DeepSeek 调用全部失败，跳过本次调优")
    return None


# ─────────────────────────────────────────────
# 参数校验与合并
# ─────────────────────────────────────────────
def validate_changes(changes: dict, style: str) -> dict:
    """
    对 AI 返回的参数变更做边界校验。
    超出合法范围的值会被 clamp 到边界，并记录警告。
    """
    bounds = STYLE_BOUNDS.get(style, STYLE_BOUNDS["balanced"])
    validated = {}

    param_bound_map = {
        "risk.max_position":    bounds["max_position"],
        "risk.stop_loss":       bounds["stop_loss"],
        "risk.take_profit":     bounds["take_profit"],
        "signals.rsi_overbought": bounds["rsi_overbought"],
        "signals.rsi_oversold":   bounds["rsi_oversold"],
    }
    # 权重之和校验稍后单独处理
    weight_keys = ["signals.macd_weight", "signals.rsi_weight", "signals.volume_weight"]

    for key, value in changes.items():
        if value is None:
            validated[key] = None
            continue

        if key in param_bound_map:
            lo, hi = param_bound_map[key]
            clamped = max(lo, min(hi, value))
            if clamped != value:
                logger.warning(f"参数 {key}={value} 超出范围 [{lo},{hi}]，已截断为 {clamped}")
            validated[key] = round(clamped, 4)
        else:
            validated[key] = value

    # 权重归一化：三个权重不为 None 时确保之和 = 1.0
    weights = {k: changes.get(k) for k in weight_keys}
    if all(w is not None for w in weights.values()):
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"权重之和 {total:.3f} ≠ 1.0，进行归一化")
            for k in weight_keys:
                validated[k] = round(weights[k] / total, 4)

    return validated


def apply_changes_to_config(config: dict, changes: dict) -> dict:
    """
    将 "risk.stop_loss": 0.04 这类点分路径写入 config dict。
    支持两级路径，例如 "risk.stop_loss"、"signals.rsi_overbought"。
    """
    for dotted_key, value in changes.items():
        if value is None:
            continue
        parts = dotted_key.split(".", 1)
        if len(parts) == 2:
            section, param = parts
            if section not in config:
                config[section] = {}
            config[section][param] = value
        else:
            config[dotted_key] = value
    return config


# ─────────────────────────────────────────────
# 主入口：夜间调优
# ─────────────────────────────────────────────
def run_nightly_optimization(days: int = 10, dry_run: bool = False) -> dict:
    """
    完整的夜间调优流程：
      1. 加载配置
      2. 查询绩效
      3. 构建 Prompt → 调用 DeepSeek
      4. 校验参数
      5. 备份旧配置 → 写入新配置
      6. 返回本次调优报告

    Args:
        days:    分析最近几天的绩效，默认 10 天
        dry_run: 为 True 时只打印建议，不实际修改文件（调试用）

    Returns:
        report dict，包含 action / reason / changes / backup_path
    """
    logger.info("=" * 50)
    logger.info("夜间策略参数调优开始")
    logger.info("=" * 50)

    # Step 1: 加载配置
    config = load_config()
    style = config.get("strategy", {}).get("style", "balanced")
    logger.info(f"当前策略风格：{style}")

    # Step 2: 查询绩效
    performance = fetch_recent_performance(days=days)
    logger.info(
        f"近 {days} 天绩效：{performance['total_signals']} 条信号，"
        f"胜率 {performance['win_rate_pct']}%，"
        f"平均收益 {performance['avg_return_pct']}%"
    )

    if performance["total_signals"] < 3:
        logger.info("信号数量不足 3 条，跳过本次调优")
        return {"action": "no_change", "reason": "信号数量不足，跳过调优", "changes": {}}

    # Step 3: 构建 Prompt 并调用 AI
    prompt = build_optimizer_prompt(config, performance)
    result = call_deepseek(prompt)

    if result is None:
        return {"action": "error", "reason": "DeepSeek API 调用失败", "changes": {}}

    action = result.get("action", "no_change")
    reason = result.get("reason", "")
    raw_changes = result.get("changes", {})

    logger.info(f"AI 建议：action={action}，reason={reason}")
    logger.info(f"AI 原始变更：{raw_changes}")

    report = {
        "action": action,
        "reason": reason,
        "changes": {},
        "backup_path": None,
        "timestamp": datetime.now().isoformat(),
        "performance_snapshot": performance,
    }

    if action != "update":
        logger.info("AI 判断无需调整参数，本次调优结束")
        return report

    # Step 4: 校验参数边界
    validated = validate_changes(raw_changes, style)
    # 过滤掉 null 值，只保留实际变动
    actual_changes = {k: v for k, v in validated.items() if v is not None}

    if not actual_changes:
        logger.info("校验后无有效变更，跳过写入")
        report["action"] = "no_change"
        report["reason"] += "（校验后无有效变更）"
        return report

    logger.info(f"校验后实际变更：{actual_changes}")
    report["changes"] = actual_changes

    if dry_run:
        logger.info("[DRY RUN] 不写入文件，仅展示变更内容")
        return report

    # Step 5: 备份旧配置 → 应用变更 → 写入
    backup_path = backup_config()
    report["backup_path"] = str(backup_path)

    new_config = apply_changes_to_config(config, actual_changes)
    save_config(new_config)

    logger.info(f"参数调优完成，共变更 {len(actual_changes)} 项")
    logger.info("=" * 50)

    return report


# ─────────────────────────────────────────────
# 工具函数：手动调优 / 回滚 / 查看历史
# ─────────────────────────────────────────────
def manual_optimize(days: int = 10) -> None:
    """手动触发调优（命令行使用）"""
    report = run_nightly_optimization(days=days, dry_run=False)
    print("\n── 调优报告 ──────────────────────────────")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def rollback_to_last_backup() -> bool:
    """
    回滚到最近一次备份的配置。
    返回 True 表示成功，False 表示无备份可用。
    """
    backups = list_config_backups()
    if not backups:
        logger.warning("没有可用的配置备份")
        return False
    latest = backups[0]
    restore_config(latest)
    logger.info(f"已回滚到备份：{latest.name}")
    return True


def show_optimization_history(n: int = 5) -> None:
    """打印最近 n 个配置备份的文件名"""
    backups = list_config_backups()[:n]
    if not backups:
        print("暂无备份记录")
        return
    print(f"\n最近 {len(backups)} 次配置备份：")
    for i, p in enumerate(backups, 1):
        stat = p.stat()
        size_kb = stat.st_size / 1024
        print(f"  {i}. {p.name}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────
# APScheduler 注册函数（供 scheduler.py 调用）
# ─────────────────────────────────────────────
def scheduled_optimize() -> None:
    """
    APScheduler 定时调用的入口，每天 22:00 自动执行。

    在 scheduler.py 中注册示例：
        from strategy.optimizer import scheduled_optimize
        scheduler.add_job(scheduled_optimize, 'cron', hour=22, minute=0)
    """
    try:
        report = run_nightly_optimization(days=10, dry_run=False)
        if report["action"] == "update":
            logger.info(f"定时调优完成，变更了 {len(report['changes'])} 个参数")
        else:
            logger.info(f"定时调优无变更：{report['reason']}")
    except Exception as e:
        logger.error(f"定时调优异常：{e}", exc_info=True)


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="策略参数夜间调优工具")
    parser.add_argument(
        "command",
        choices=["optimize", "rollback", "history", "dry-run"],
        help=(
            "optimize: 执行调优并写入配置 | "
            "rollback: 回滚到上一版本 | "
            "history: 查看备份历史 | "
            "dry-run: 只输出建议不写入"
        ),
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="分析最近几天的绩效（默认 10）"
    )
    args = parser.parse_args()

    if args.command == "optimize":
        manual_optimize(days=args.days)

    elif args.command == "dry-run":
        report = run_nightly_optimization(days=args.days, dry_run=True)
        print("\n── [DRY RUN] 调优建议 ────────────────────")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    elif args.command == "rollback":
        success = rollback_to_last_backup()
        print("✓ 回滚成功" if success else "✗ 无可用备份")

    elif args.command == "history":
        show_optimization_history(n=10)
