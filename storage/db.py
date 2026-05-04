"""
storage/db.py - SQLite 持久化存储（统一数据访问层）
所有模块通过此模块操作数据库，统一 DB_PATH = db/quant.db
"""
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "db" / "quant.db"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_signals_table():
    """确保 signals 表按 flat schema 创建（幂等）"""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol            TEXT    NOT NULL,
            signal_date       TEXT    NOT NULL,
            direction         TEXT,
            confidence        INTEGER,
            suggested_position REAL,
            stop_loss         REAL,
            reasoning         TEXT,
            composite_score   REAL,
            actual_return     REAL    DEFAULT NULL,
            is_correct        INTEGER DEFAULT NULL,
            UNIQUE(symbol, signal_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            sharpe REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_return REAL,
            n_trades INTEGER,
            config_version INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            config_json TEXT NOT NULL,
            change_reason TEXT
        )
    """)
    conn.commit()
    conn.close()


def init_db():
    """兼容接口：初始化所有表"""
    ensure_signals_table()


# ══════════════════════════════════════════════
# 信号读写
# ══════════════════════════════════════════════

def save_signal(symbol: str, signals: dict, analysis: dict) -> bool:
    """
    scheduler.py 兼容接口：将 signals + analysis 写入 signals 表。
    signals = {composite, macd, rsi_value, ...} 技术因子字典
    analysis = {direction, confidence, suggested_position, ...} AI 决策
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return _write_signal({
        "symbol": symbol,
        "signal_date": today,
        "direction": analysis.get("direction", "hold"),
        "confidence": int((analysis.get("confidence") or 0.5) * 100),
        "suggested_position": analysis.get("suggested_position", 0.0),
        "stop_loss": analysis.get("stop_loss_pct", 5.0) / 100 if analysis.get("stop_loss_pct") else 0.05,
        "reasoning": analysis.get("reasoning", ""),
        "composite_score": signals.get("composite", 0),
    })


def _write_signal(signal: dict) -> bool:
    try:
        conn = get_conn()
        conn.execute("""
            INSERT INTO signals
                (symbol, signal_date, direction, confidence,
                 suggested_position, stop_loss, reasoning, composite_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, signal_date) DO UPDATE SET
                direction          = excluded.direction,
                confidence         = excluded.confidence,
                suggested_position = excluded.suggested_position,
                stop_loss          = excluded.stop_loss,
                reasoning          = excluded.reasoning,
                composite_score    = excluded.composite_score
        """, (
            signal["symbol"], signal["signal_date"],
            signal.get("direction", "hold"),
            signal.get("confidence", 50),
            signal.get("suggested_position", 0.0),
            signal.get("stop_loss", 0.05),
            signal.get("reasoning", ""),
            signal.get("composite_score", 0),
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"写入信号失败: {e}")
        return False


def get_latest_signals(n: int = 10) -> list[dict]:
    """获取最新 N 条信号（跨所有标的）"""
    if not DB_PATH.exists():
        return []
    try:
        conn = get_conn()
        cur = conn.execute(
            """SELECT symbol, signal_date, direction, confidence,
                      suggested_position, stop_loss, actual_return, is_correct,
                      composite_score, reasoning
               FROM signals
               ORDER BY signal_date DESC, id DESC
               LIMIT ?""",
            (n,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except sqlite3.Error as e:
        logger.error(f"查询信号失败: {e}")
        return []


# ══════════════════════════════════════════════
# 绩效
# ══════════════════════════════════════════════

def save_performance(symbol: str, perf: dict, config_version: int = 1):
    """写入日绩效"""
    try:
        conn = get_conn()
        conn.execute(
            """INSERT INTO performance
               (symbol, date, sharpe, max_drawdown, win_rate, total_return, n_trades, config_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, datetime.now().strftime("%Y-%m-%d"),
             perf.get("sharpe"), perf.get("max_drawdown"),
             perf.get("win_rate"), perf.get("total_return"),
             perf.get("n_trades"), config_version),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"写入绩效失败: {e}")


def get_performance_history(symbol: str, days: int = 30) -> list:
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM performance WHERE symbol=? ORDER BY date DESC LIMIT ?",
            (symbol, days),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_performance_summary(days: int = 30) -> dict:
    """近 N 天绩效摘要"""
    if not DB_PATH.exists():
        return {}
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        conn = get_conn()
        cur = conn.execute(
            """SELECT COUNT(*) AS total,
                      AVG(CASE WHEN is_correct=1 THEN 1.0 ELSE 0.0 END) AS win_rate,
                      AVG(actual_return) AS avg_return,
                      MIN(actual_return) AS worst,
                      MAX(actual_return) AS best
               FROM signals
               WHERE signal_date >= ? AND actual_return IS NOT NULL""",
            (cutoff,),
        )
        result = dict(cur.fetchone() or {})
        conn.close()
        return result
    except Exception as e:
        logger.error(f"查询绩效失败: {e}")
        return {}


def get_equity_curve(days: int = 60) -> pd.DataFrame:
    """近 N 天累计净值曲线"""
    if not DB_PATH.exists():
        return pd.DataFrame()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        conn = get_conn()
        df = pd.read_sql_query(
            """SELECT signal_date as date, actual_return
               FROM signals
               WHERE signal_date >= ? AND actual_return IS NOT NULL
               ORDER BY signal_date""",
            conn, params=(cutoff,),
        )
        conn.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date")["actual_return"].mean()
        equity = (1 + daily).cumprod()
        return equity.reset_index().rename(columns={"actual_return": "equity"})
    except Exception as e:
        logger.error(f"获取权益曲线失败: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════
# 配置版本
# ══════════════════════════════════════════════

def save_config_version(config: dict, reason: str = "手动更新"):
    import json
    try:
        conn = get_conn()
        version = config.get("version", 1)
        conn.execute(
            "INSERT INTO config_versions (version, timestamp, config_json, change_reason) VALUES (?, ?, ?, ?)",
            (version, datetime.now().isoformat(), json.dumps(config, ensure_ascii=False), reason),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"保存配置版本失败: {e}")


# ══════════════════════════════════════════════
# 监控状态
# ══════════════════════════════════════════════

def get_db_status() -> dict:
    """返回数据库概览状态"""
    status = {}
    try:
        conn = get_conn()

        # 各表行数
        for table in ["signals", "market_data", "northbound_flow",
                       "sentiment_index", "margin_data", "futures_basis"]:
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                status[f"{table}_rows"] = cur.fetchone()[0]
            except Exception:
                status[f"{table}_rows"] = 0

        # 信号时间范围
        cur = conn.execute("SELECT MAX(signal_date), MIN(signal_date) FROM signals")
        row = cur.fetchone()
        if row and row[0]:
            status["signal_latest"] = row[0]
            status["signal_oldest"] = row[1]
            status["signal_total"] = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            status["signal_settled"] = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE actual_return IS NOT NULL").fetchone()[0]

        # 行情时间范围
        cur = conn.execute("SELECT MAX(date), MIN(date) FROM market_data")
        row = cur.fetchone()
        if row and row[0]:
            status["market_latest"] = row[0]
            status["market_oldest"] = row[1]

        # 北向资金
        cur = conn.execute("SELECT MAX(date) FROM northbound_flow WHERE net_flow IS NOT NULL")
        row = cur.fetchone()
        if row and row[0]:
            status["northbound_latest"] = row[0]

        # 融资融券
        cur = conn.execute("SELECT MAX(date) FROM margin_data")
        row = cur.fetchone()
        if row and row[0]:
            status["margin_latest"] = row[0]

        # 期货升贴水
        cur = conn.execute("SELECT MAX(date) FROM futures_basis")
        row = cur.fetchone()
        if row and row[0]:
            status["futures_latest"] = row[0]

        # 情绪
        cur = conn.execute(
            "SELECT value, label FROM sentiment_index ORDER BY date DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            status["sentiment_value"] = row[0]
            status["sentiment_label"] = row[1]

        # 数据库大小
        status["db_size_mb"] = round(DB_PATH.stat().st_size / (1024 * 1024), 2) if DB_PATH.exists() else 0

        conn.close()
    except Exception as e:
        logger.error(f"查询数据库状态失败: {e}")

    return status
