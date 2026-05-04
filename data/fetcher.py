# -*- coding: utf-8 -*-
"""
data/fetcher.py
A股行情数据拉取模块

功能：
  - 从 AKShare 拉取 A股/ETF/指数 日线行情
  - 增量更新：只拉取数据库中没有的日期，不重复下载
  - 自动跳过非交易日（周末）
  - 拉取失败自动重试，单个标的失败不影响其他标的
  - 支持手动触发和 APScheduler 定时调用

数据库表（自动创建）：
  market_data(symbol, date, open, high, low, close, volume, amount, pct_change)
  fetch_log(id, symbol, fetch_time, status, rows_added, message)

依赖：
  pip install akshare pandas pyyaml

用法：
  python data/fetcher.py                   # 拉取 config.yaml 中所有标的
  python data/fetcher.py --symbol 000300   # 只拉取指定标的
  python data/fetcher.py --days 30         # 强制拉取最近 N 天
  python data/fetcher.py --status          # 查看数据库状态
"""

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────
# 路径 & 日志（目录先建，再初始化日志）
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT_DIR / "db" / "quant.db"
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "fetcher.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetcher")


# ─────────────────────────────────────────────
# 标的类型识别
# ─────────────────────────────────────────────
# 常见指数代码集合
_INDEX_CODES = {
    "000001", "000300", "000905", "000016", "000010", "000009",
    "399001", "399006", "399300", "399905",
}

def _symbol_type(symbol: str) -> str:
    """判断标的类型：index / etf / stock"""
    s = str(symbol).strip()
    if s in _INDEX_CODES:
        return "index"
    # ETF：5 或 15/16 开头
    if s.startswith("5") or s.startswith("15") or s.startswith("16"):
        return "etf"
    return "stock"


# ─────────────────────────────────────────────
# 数据库初始化
# ─────────────────────────────────────────────
def init_db() -> None:
    """创建数据库表（如不存在则新建）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            symbol      TEXT NOT NULL,
            date        TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            amount      REAL,
            pct_change  REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            fetch_time  TEXT,
            status      TEXT,
            rows_added  INTEGER DEFAULT 0,
            message     TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"数据库就绪：{DB_PATH}")


def _log_fetch(symbol: str, status: str, rows_added: int, message: str = "") -> None:
    """写入拉取日志到 fetch_log 表"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO fetch_log (symbol, fetch_time, status, rows_added, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, datetime.now().isoformat(), status, rows_added, message),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logger.warning(f"写入拉取日志失败：{e}")


# ─────────────────────────────────────────────
# 获取数据库中最新日期
# ─────────────────────────────────────────────
def get_latest_db_date(symbol: str) -> str | None:
    """返回该标的在数据库中最新的 date 字符串，无数据返回 None"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT MAX(date) FROM market_data WHERE symbol = ?", (symbol,)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except sqlite3.Error:
        return None


# ─────────────────────────────────────────────
# AKShare 拉取（三种接口）
# ─────────────────────────────────────────────
def _fetch_index(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取指数行情"""
    import akshare as ak
    df = ak.stock_zh_index_daily_em(symbol=symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["amount"]     = None
    df["pct_change"] = df["close"].pct_change() * 100
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return df.reset_index(drop=True)


def _fetch_etf(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 ETF 行情（前复权）"""
    import akshare as ak
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    col_map = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
               "收盘": "close", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def _fetch_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉取个股行情（前复权）"""
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    col_map = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
               "收盘": "close", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 写入数据库（INSERT OR IGNORE 防重复）
# ─────────────────────────────────────────────
def _save_to_db(symbol: str, df: pd.DataFrame) -> int:
    """
    将 DataFrame 写入 market_data 表。
    已存在的 (symbol, date) 主键自动跳过。
    返回实际新增的行数。
    """
    if df.empty:
        return 0

    # 补全缺失列
    for col in ["amount", "pct_change"]:
        if col not in df.columns:
            df[col] = None

    conn = sqlite3.connect(DB_PATH)
    rows_before = conn.execute(
        "SELECT COUNT(*) FROM market_data WHERE symbol = ?", (symbol,)
    ).fetchone()[0]

    for _, row in df.iterrows():
        conn.execute(
            """
            INSERT OR IGNORE INTO market_data
                (symbol, date, open, high, low, close, volume, amount, pct_change)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                str(row.get("date", "")),
                _safe_float(row.get("open")),
                _safe_float(row.get("high")),
                _safe_float(row.get("low")),
                _safe_float(row.get("close")),
                _safe_float(row.get("volume")),
                _safe_float(row.get("amount")),
                _safe_float(row.get("pct_change")),
            ),
        )
    conn.commit()

    rows_after = conn.execute(
        "SELECT COUNT(*) FROM market_data WHERE symbol = ?", (symbol,)
    ).fetchone()[0]
    conn.close()
    return rows_after - rows_before


def _safe_float(val) -> float | None:
    """安全转换为 float，异常返回 None"""
    try:
        return float(val) if val is not None and str(val) not in ("", "nan", "NaN") else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# 单标的拉取（含重试）
# ─────────────────────────────────────────────
def fetch_symbol(
    symbol: str,
    force_days: int | None = None,
    max_retries: int = 3,
) -> int:
    """
    拉取单个标的最新行情，增量写入数据库。

    Args:
        symbol:      标的代码，如 "000300"
        force_days:  强制拉取最近 N 天（不管数据库是否已有）
        max_retries: API 失败时重试次数

    Returns:
        新写入行数（≥0），失败返回 -1
    """
    today = datetime.now().strftime("%Y-%m-%d")
    stype = _symbol_type(symbol)

    # 确定起始日期
    if force_days:
        start_date = (datetime.now() - timedelta(days=force_days)).strftime("%Y-%m-%d")
    else:
        latest = get_latest_db_date(symbol)
        if latest:
            start_date = (
                datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    if start_date > today:
        logger.info(f"{symbol} 数据已是最新，跳过")
        return 0

    logger.info(f"拉取 {symbol}（{stype}）：{start_date} → {today}")

    df          = pd.DataFrame()
    last_error  = ""

    for attempt in range(1, max_retries + 1):
        try:
            if stype == "index":
                df = _fetch_index(symbol, start_date, today)
            elif stype == "etf":
                df = _fetch_etf(symbol, start_date, today)
            else:
                df = _fetch_stock(symbol, start_date, today)

            if df is not None and not df.empty:
                break   # 成功

        except Exception as e:
            last_error = str(e)
            logger.warning(f"{symbol} 第{attempt}次拉取失败：{e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)

    if df is None or df.empty:
        msg = f"数据为空：{last_error}" if last_error else "数据为空（可能是非交易日）"
        logger.warning(f"{symbol} {msg}")
        _log_fetch(symbol, "empty", 0, msg)
        return 0

    try:
        rows_added = _save_to_db(symbol, df)
        logger.info(f"{symbol} 新增 {rows_added} 条")
        _log_fetch(symbol, "ok", rows_added)
        return rows_added
    except Exception as e:
        logger.error(f"{symbol} 写入数据库失败：{e}")
        _log_fetch(symbol, "db_error", 0, str(e))
        return -1


# ─────────────────────────────────────────────
# 批量拉取（读取 config.yaml）
# ─────────────────────────────────────────────
def _load_symbols() -> list:
    """从 config.yaml 读取标的列表"""
    try:
        import yaml
        cfg_path = ROOT_DIR / "config.yaml"
        if not cfg_path.exists():
            return ["000300", "510050"]
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        symbols = cfg.get("strategy", {}).get("symbols", ["000300", "510050"])
        return [str(s) for s in symbols]
    except Exception as e:
        logger.error(f"读取配置失败：{e}")
        return ["000300", "510050"]


def update_all(force_days: int | None = None) -> dict:
    """
    拉取 config.yaml 中全部标的的最新行情。

    Args:
        force_days: 强制拉取最近 N 天，None 表示增量更新

    Returns:
        {symbol: rows_added}，rows_added 为 -1 表示失败
    """
    init_db()
    symbols = _load_symbols()
    results = {}

    logger.info(f"开始批量拉取，共 {len(symbols)} 个标的")

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}")
        results[symbol] = fetch_symbol(symbol, force_days=force_days)
        if i < len(symbols):
            time.sleep(1.0)   # 避免触发频率限制

    success    = sum(1 for v in results.values() if v >= 0)
    total_rows = sum(v for v in results.values() if v > 0)
    logger.info(f"批量拉取完成：{success}/{len(symbols)} 成功，共新增 {total_rows} 条")
    return results


# ─────────────────────────────────────────────
# 数据查询（供 signals.py 调用）
# ─────────────────────────────────────────────
def get_ohlcv(symbol: str, days: int = 120) -> pd.DataFrame:
    """
    读取指定标的最近 N 天的 OHLCV 数据，按日期升序返回。

    Returns:
        DataFrame，列：date(datetime), open, high, low, close, volume, amount, pct_change
        无数据时返回空 DataFrame
    """
    if not DB_PATH.exists():
        logger.warning("数据库不存在，请先运行 update_all()")
        return pd.DataFrame()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            """
            SELECT date, open, high, low, close, volume, amount, pct_change
            FROM market_data
            WHERE symbol = ? AND date >= ?
            ORDER BY date ASC
            """,
            conn, params=(symbol, cutoff),
        )
        conn.close()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except sqlite3.Error as e:
        logger.error(f"读取 {symbol} 数据失败：{e}")
        return pd.DataFrame()


def get_latest_close(symbol: str) -> float | None:
    """获取标的最新收盘价，无数据返回 None"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT close FROM market_data WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        conn.close()
        return float(row[0]) if row else None
    except sqlite3.Error:
        return None


def get_db_status() -> list:
    """返回数据库中每个标的的数据概况，用于诊断"""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT symbol, COUNT(*) AS cnt, MAX(date) AS latest, MIN(date) AS oldest
            FROM market_data GROUP BY symbol ORDER BY symbol
        """).fetchall()
        conn.close()
        return [
            {"symbol": r[0], "count": r[1], "latest_date": r[2], "oldest_date": r[3]}
            for r in rows
        ]
    except sqlite3.Error as e:
        logger.error(f"查询数据状态失败：{e}")
        return []


# ─────────────────────────────────────────────
# APScheduler 注册函数（供 scheduler.py 调用）
# ─────────────────────────────────────────────
def scheduled_fetch() -> None:
    """
    每日 09:00 自动触发，在 scheduler.py 中注册：

        from data.fetcher import scheduled_fetch
        scheduler.add_job(scheduled_fetch, 'cron', hour=9, minute=0)
    """
    try:
        results = update_all()
        ok    = sum(1 for v in results.values() if v >= 0)
        added = sum(v for v in results.values() if v > 0)
        logger.info(f"定时拉取完成：{ok} 个标的成功，新增 {added} 条")
    except Exception as e:
        logger.error(f"定时拉取异常：{e}", exc_info=True)


# ─────────────────────────────────────────────
# 北向资金数据
# ─────────────────────────────────────────────
def init_northbound_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS northbound_flow (
            date            TEXT PRIMARY KEY,
            net_flow        REAL,
            buy_amount      REAL,
            sell_amount     REAL,
            balance         REAL,
            cumulative_flow REAL
        )
    """)
    conn.commit()
    conn.close()


def fetch_northbound_flow(days: int = 0) -> int:
    """
    从 AKShare 拉取北向资金每日净流入数据。
    注：AKShare 该接口数据目前更新到 2024-08，之后的数据需等接口修复。
    Args:
        days: 保留参数，0 表示拉取全部可用历史数据
    Returns: 新增行数，-1 表示失败。
    """
    try:
        import akshare as ak
        import numpy as np
    except ImportError:
        logger.warning("AKShare 未安装，跳过北向资金拉取")
        return -1

    init_northbound_db()

    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
    except Exception as e:
        logger.error(f"拉取北向资金数据失败: {e}")
        _log_fetch("northbound", "error", 0, str(e))
        return -1

    if df is None or df.empty:
        logger.warning("北向资金数据为空")
        return 0

    # 列名映射：日期/当日成交净买额/买入成交额/卖出成交额/资金余额/历史累计净买额
    col_map = {
        "日期": "date", "当日成交净买额": "net_flow",
        "买入成交额": "buy_amount", "卖出成交额": "sell_amount",
        "当日资金余额": "balance", "历史累计净买额": "cumulative_flow",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 只保留有实际成交数据（net_flow 非 NaN）的行
    df = df[df["net_flow"].notna()].copy()
    if df.empty:
        logger.warning("北向资金无有效数据行")
        return 0

    # 不按天数过滤，全部入库（INSERT OR IGNORE 自动去重）
    conn = sqlite3.connect(DB_PATH)
    added = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                """INSERT OR REPLACE INTO northbound_flow
                   (date, net_flow, buy_amount, sell_amount, balance, cumulative_flow)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(row["date"]),
                    _safe_float(row.get("net_flow")),
                    _safe_float(row.get("buy_amount")),
                    _safe_float(row.get("sell_amount")),
                    _safe_float(row.get("balance")),
                    _safe_float(row.get("cumulative_flow")),
                ),
            )
            if conn.changes > 0:
                added += 1
        except Exception:
            pass
    conn.commit()
    conn.close()

    logger.info(f"北向资金：新增 {added} 条（总计 {added} 条有效数据）")
    _log_fetch("northbound", "ok" if added >= 0 else "empty", added)
    return added


def get_northbound_flow(days: int = 60) -> pd.DataFrame:
    """查询北向资金数据，返回最近 N 个交易日的 DataFrame"""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        # 注：AKShare 数据截止 2024-08-16，所以用 ORDER BY date DESC LIMIT 而非日期过滤
        df = pd.read_sql_query(
            "SELECT * FROM northbound_flow WHERE net_flow IS NOT NULL ORDER BY date DESC LIMIT ?",
            conn, params=(days,),
        )
        conn.close()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)  # 按日期升序（图表从左到右）
        return df
    except Exception as e:
        logger.error(f"读取北向资金数据失败: {e}")
        return pd.DataFrame()


def get_latest_northbound() -> dict:
    """获取最近一个交易日的北向资金摘要"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT * FROM northbound_flow ORDER BY date DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return {
                "date": row[0],
                "net_flow": row[1],
                "buy_amount": row[2],
                "sell_amount": row[3],
                "balance": row[4],
                "cumulative_flow": row[5],
            }
        return {}
    except Exception:
        return {}


# ─────────────────────────────────────────────
# 市场情绪指数
# ─────────────────────────────────────────────
def init_sentiment_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_index (
            date              TEXT PRIMARY KEY,
            value             REAL,
            label             TEXT,
            northbound_score  REAL,
            fund_flow_score   REAL,
            breadth_score     REAL,
            momentum_score    REAL,
            margin_score      REAL,
            basis_score       REAL
        )
    """)
    # 兼容旧表：如果列不存在则添加
    for col in ["margin_score", "basis_score"]:
        try:
            conn.execute(f"ALTER TABLE sentiment_index ADD COLUMN {col} REAL")
        except Exception:
            pass
    conn.commit()
    conn.close()


def _calc_sentiment_score(nb_df: pd.DataFrame, config: dict) -> dict | None:
    """
    基于多维度数据计算综合市场情绪 0-100 分。
    维度（权重由 config.yaml 控制）：
      1. 融资情绪     — 融资余额变化，杠杆资金态度
      2. 主力资金流向 — 主力净流入方向
      3. 期货升贴水   — IF 基差，机构情绪
      4. 市场宽度     — 涨跌比
      5. 指数动量     — 价格 vs MA20 偏离
    """
    sent_cfg = config.get("sentiment", {})
    w_margin = sent_cfg.get("weights", {}).get("margin", 0.30)
    w_fund   = sent_cfg.get("weights", {}).get("fund_flow", 0.25)
    w_basis  = sent_cfg.get("weights", {}).get("futures_basis", 0.20)
    w_bread  = sent_cfg.get("weights", {}).get("breadth", 0.15)
    w_mom    = sent_cfg.get("weights", {}).get("momentum", 0.10)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── 1. 融资情绪 ──────────────────────────
    margin_score = 50.0
    try:
        df = get_margin_data(days=10)
        if not df.empty and "total_margin" in df.columns:
            latest = float(df["total_margin"].iloc[-1])
            prev   = float(df["total_margin"].iloc[0])
            if prev > 0:
                change_pct = (latest - prev) / prev * 100
                # 10日融资余额变化百分比：> +5% 看多，< -5% 看空
                margin_score = 50 + min(25, max(-25, change_pct * 5))
    except Exception as e:
        logger.debug(f"融资情绪计算失败: {e}")
    margin_score = round(max(0, min(100, margin_score)), 2)

    # ── 2. 主力资金流向得分 ──────────────────
    fund_score = 50.0
    try:
        import akshare as ak
        fund_df = ak.stock_market_fund_flow()
        if fund_df is not None and not fund_df.empty:
            main_col = None
            for c in fund_df.columns:
                if "主力" in str(c) and "净" in str(c):
                    main_col = c
                    break
            if main_col is not None and len(fund_df) > 0:
                main_flow = float(fund_df.iloc[0][main_col])
                fund_score = 50 + min(30, max(-30, main_flow / 10))
    except Exception as e:
        logger.debug(f"主力资金流拉取失败: {e}")
    fund_score = round(max(0, min(100, fund_score)), 2)

    # ── 3. 期货升贴水得分 ────────────────────
    basis_score = 50.0
    try:
        df = get_futures_basis(days=5)
        if not df.empty and "basis_pct" in df.columns:
            avg_basis = float(df["basis_pct"].mean())
            # 基差百分比：> +0.5% 升水看多，< -0.5% 贴水看空
            basis_score = 50 + min(25, max(-25, avg_basis * 30))
    except Exception as e:
        logger.debug(f"期货升贴水计算失败: {e}")
    basis_score = round(max(0, min(100, basis_score)), 2)

    # ── 4. 市场宽度得分 ──────────────────────
    breadth_score = 50.0
    try:
        import akshare as ak
        sh_df = ak.stock_zh_index_daily_em(symbol="sh000001")
        if sh_df is not None and not sh_df.empty:
            sh_df = sh_df.tail(5)
            pct_changes = pd.to_numeric(sh_df.iloc[:, 1], errors="coerce")
            up_days = (pct_changes > 0).sum()
            breadth_score = up_days / len(pct_changes) * 100
    except Exception as e:
        logger.debug(f"市场宽度计算失败: {e}")
    breadth_score = round(max(0, min(100, breadth_score)), 2)

    # ── 5. 指数动量得分 ──────────────────────
    momentum_score = 50.0
    try:
        import akshare as ak
        sh_df = ak.stock_zh_index_daily_em(symbol="sh000001")
        if sh_df is not None and not sh_df.empty and len(sh_df) >= 20:
            closes = pd.to_numeric(sh_df.iloc[:, 1], errors="coerce")
            ma20   = closes.rolling(20).mean()
            latest = float(closes.iloc[-1])
            ma20_v = float(ma20.iloc[-1])
            if ma20_v > 0:
                dev_pct = (latest - ma20_v) / ma20_v * 100
                momentum_score = 50 + dev_pct * 10
    except Exception as e:
        logger.debug(f"动量得分计算失败: {e}")
    momentum_score = round(max(0, min(100, momentum_score)), 2)

    # ── 加权综合 ─────────────────────────────
    value = (w_margin * margin_score + w_fund * fund_score +
             w_basis * basis_score + w_bread * breadth_score +
             w_mom * momentum_score)
    value = round(max(0, min(100, value)), 2)

    extreme_fear = sent_cfg.get("extreme_fear", 25)
    fear        = sent_cfg.get("fear", 45)
    neutral     = sent_cfg.get("neutral", 55)
    greed       = sent_cfg.get("greed", 75)

    if value <= extreme_fear:
        label = "extreme_fear"
    elif value <= fear:
        label = "fear"
    elif value <= neutral:
        label = "neutral"
    elif value <= greed:
        label = "greed"
    else:
        label = "extreme_greed"

    return {
        "date": today_str,
        "value": value,
        "label": label,
        "margin_score": margin_score,
        "fund_flow_score": fund_score,
        "basis_score": basis_score,
        "breadth_score": breadth_score,
        "momentum_score": momentum_score,
    }


# ── 情绪指数缓存（TTL = 30 分钟，API 调用不阻塞）──
import threading as _threading
_sentiment_cache = {"data": None, "ts": 0}
_sentiment_lock = _threading.Lock()
_SENTIMENT_TTL = 1800  # 30 秒

def _refresh_sentiment_background(config: dict):
    """后台刷新情绪数据"""
    try:
        _calc_and_save_sentiment(config)
        _sentiment_cache["data"] = _fetch_latest_sentiment_from_db()
        _sentiment_cache["ts"] = datetime.now().timestamp()
        logger.info("情绪缓存后台刷新完成")
    except Exception as e:
        logger.error(f"情绪缓存后台刷新失败: {e}")


def _calc_and_save_sentiment(config: dict) -> dict | None:
    """计算情绪并写入 DB（原 fetch_fear_greed_index 主体逻辑）"""
    init_sentiment_db()
    fetch_margin_data()
    fetch_futures_basis()
    nb_weight = config.get("sentiment", {}).get("weights", {}).get("northbound", 0)
    if nb_weight > 0:
        fetch_northbound_flow(days=60)
        nb_df = get_northbound_flow(days=60)
    else:
        nb_df = pd.DataFrame()

    result = _calc_sentiment_score(nb_df, config)
    if result is None:
        return None

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT OR REPLACE INTO sentiment_index
               (date, value, label, northbound_score, fund_flow_score, breadth_score,
                momentum_score, margin_score, basis_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result["date"], result["value"], result["label"],
                result.get("northbound_score", 50), result.get("fund_flow_score", 50),
                result.get("breadth_score", 50), result.get("momentum_score", 50),
                result.get("margin_score", 50), result.get("basis_score", 50),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"写入情绪数据失败: {e}")

    return result


def _fetch_latest_sentiment_from_db() -> dict:
    """直接从 DB 读最新情绪"""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT * FROM sentiment_index ORDER BY date DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            return {k: row[i] for i, k in enumerate(
                ["date", "value", "label", "northbound_score", "fund_flow_score",
                 "breadth_score", "momentum_score", "margin_score", "basis_score"]
            ) if i < len(row)}
        return {}
    except Exception:
        return {}


def fetch_fear_greed_index(config: dict = None) -> dict | None:
    """
    综合市场情绪指数，带 30 分钟 TTL 缓存。
    首次调用触发异步后台刷新，后续调用直接返回缓存。
    """
    now = datetime.now().timestamp()

    # 检查缓存是否有效
    with _sentiment_lock:
        if _sentiment_cache["data"] and (now - _sentiment_cache["ts"]) < _SENTIMENT_TTL:
            return _sentiment_cache["data"]
        stale_data = _sentiment_cache["data"]

    # 尝试同步刷新
    try:
        if config is None:
            import yaml
            cfg_path = ROOT_DIR / "config.yaml"
            config = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) if cfg_path.exists() else {}

        result = _calc_and_save_sentiment(config)
        if result:
            with _sentiment_lock:
                _sentiment_cache["data"] = result
                _sentiment_cache["ts"] = now
            logger.info(f"情绪已计算: {result['value']:.1f} ({result['label']})")
            return result
    except Exception as e:
        logger.error(f"情绪计算失败: {e}")

    # 失败时返回 stale data 兜底
    return stale_data or _fetch_latest_sentiment_from_db() or         {"value": 50, "label": "neutral", "date": datetime.now().strftime("%Y-%m-%d"),
         "margin_score": 50, "fund_flow_score": 50, "basis_score": 50,
         "breadth_score": 50, "momentum_score": 50}# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股行情数据拉取工具")
    parser.add_argument("--symbol", type=str, default=None,
                        help="指定标的代码（不填则拉取 config.yaml 中所有标的）")
    parser.add_argument("--days",   type=int, default=None,
                        help="强制拉取最近 N 天（不填则增量更新）")
    parser.add_argument("--status", action="store_true",
                        help="查看数据库中各标的的数据状态")
    parser.add_argument("--northbound", action="store_true",
                        help="拉取北向资金数据")
    parser.add_argument("--sentiment", action="store_true",
                        help="计算并存入市场情绪指数")
    args = parser.parse_args()

    init_db()

    if args.status:
        rows = get_db_status()
        if not rows:
            print("数据库为空，尚无数据")
        else:
            print(f"\n{'标的':<12} {'条数':<8} {'最新日期':<14} {'最早日期'}")
            print("-" * 52)
            for r in rows:
                print(f"{r['symbol']:<12} {r['count']:<8} {r['latest_date']:<14} {r['oldest_date']}")
        # 也显示北向和情绪
        nb = get_latest_northbound()
        sent = get_latest_sentiment()
        if nb:
            print(f"\n北向资金: {nb['date']}  净流入 {nb.get('net_flow', 0):.2f} 亿")
        if sent:
            label_map = {
                "extreme_fear": "极度恐惧", "fear": "恐惧", "neutral": "中性",
                "greed": "贪婪", "extreme_greed": "极度贪婪",
            }
            print(f"市场情绪: {sent['value']:.1f} ({label_map.get(sent.get('label', ''), sent.get('label', ''))})")

    elif args.northbound:
        n = fetch_northbound_flow(days=args.days or 60)
        print(f"{'✅' if n >= 0 else '❌'} 北向资金 新增 {n} 条")

    elif args.sentiment:
        result = fetch_fear_greed_index()
        if result:
            print(f"✅ 市场情绪: {result['value']:.1f} ({result['label']})")
        else:
            print("❌ 市场情绪计算失败")

    elif args.symbol:
        n = fetch_symbol(args.symbol, force_days=args.days)
        print(f"{'✅' if n >= 0 else '❌'} {args.symbol} 新增 {n} 条" if n >= 0
              else f"❌ {args.symbol} 拉取失败，请查看 logs/fetcher.log")

    else:
        results = update_all(force_days=args.days)
        print("\n── 拉取结果 ────────────────────────────")
        for sym, cnt in results.items():
            label = f"新增 {cnt} 条" if cnt > 0 else ("已是最新" if cnt == 0 else "失败")
            print(f"  {'✅' if cnt >= 0 else '❌'} {sym:<12} {label}")
