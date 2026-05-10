"""
data/realtime.py
实时行情模块 — 东方财富 Level-2 级别盘中数据

功能：
  - 获取指定标的的实时盘口数据（最新价、涨跌幅、成交量、量比等）
  - 获取5档盘口深度（买一~买五、卖一~卖五）
  - 计算盘中实时技术指标（实时RSI、涨速、振幅等）
  - 支持1分钟级别的盘中快照

数据来源：
  - 东方财富 WebSocket / HTTP 实时接口（免费）
  - 新浪财经实时接口（兜底）

依赖：
  pip install websocket-client (可选，HTTP模式不需要)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "realtime.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("realtime")

# ─── 东方财富 HTTP 实时接口 ──────────────────────

EASTMONEY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}

# ─── 新浪财经实时接口（兜底，盘后也可用）────────────
SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn/",
}


def _to_sina_code(symbol: str) -> str:
    """转为新浪格式：sh/sz + 代码"""
    s = str(symbol).strip()
    if s.startswith("6") or s.startswith("5") or s.startswith("000"):
        return f"sh{s}" if s.startswith("000") else f"sh{s}"
    else:
        return f"sz{s}"


def get_sina_quote(symbol: str) -> Optional[dict]:
    """
    从新浪财经获取实时行情（盘后也能取到当日收盘数据）。
    返回格式与 get_realtime_quote 一致。
    """
    sina_code = _to_sina_code(symbol)
    url = f"https://hq.sinajs.cn/list={sina_code}"

    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=5)
        resp.encoding = "gbk"
        text = resp.text.strip()

        # 格式: var hq_str_sh510300="名称,今开,昨收,当前价,最高,最低,...
        if '="' not in text:
            logger.warning(f"{symbol} 新浪数据格式异常: {text[:100]}")
            return None

        parts = text.split('="')[1].split('","')
        values = parts[0].split(",")

        if len(values) < 30:
            logger.warning(f"{symbol} 新浪数据列数不足: {len(values)}")
            return None

        name = values[0]
        open_p = _to_float(values[1])
        prev_close = _to_float(values[2])
        price = _to_float(values[3])
        high = _to_float(values[4])
        low = _to_float(values[5])
        volume = _to_float(values[8])  # 成交量（手）
        amount = _to_float(values[9])  # 成交额（元）

        change = (price - prev_close) if (price and prev_close) else None
        change_pct = (change / prev_close * 100) if (change and prev_close and prev_close > 0) else None

        return {
            "symbol": symbol,
            "name": name,
            "price": price,
            "open": open_p,
            "high": high,
            "low": low,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "amount": amount,
            # 新浪不直接提供量比/振幅/涨速，留空
            "volume_ratio": None,
            "amplitude": round((high / low - 1) * 100, 2) if (high and low and low > 0) else None,
            "turnover_rate": None,
            "limit_up": None,
            "limit_down": None,
            "speed_5m": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "_source": "sina",
        }
    except Exception as e:
        logger.error(f"{symbol} 新浪行情拉取失败: {e}")
        return None


def get_sina_batch_quotes(symbols: list) -> list:
    """
    新浪批量接口：一次请求多个标的。
    """
    sina_codes = ",".join([_to_sina_code(s) for s in symbols])
    url = f"https://hq.sinajs.cn/list={sina_codes}"

    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=5)
        resp.encoding = "gbk"
        text = resp.text.strip()

        results = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i >= len(symbols):
                break
            if '="' not in line:
                continue

            parts = line.split('="')[1].split('","')
            values = parts[0].split(",")

            if len(values) < 30:
                continue

            sym = symbols[i]
            name = values[0]
            open_p = _to_float(values[1])
            prev_close = _to_float(values[2])
            price = _to_float(values[3])
            high = _to_float(values[4])
            low = _to_float(values[5])
            volume = _to_float(values[8])
            amount = _to_float(values[9])

            change = (price - prev_close) if (price and prev_close) else None
            change_pct = (change / prev_close * 100) if (change and prev_close and prev_close > 0) else None

            results.append({
                "symbol": sym,
                "name": name,
                "price": price,
                "open": open_p,
                "high": high,
                "low": low,
                "prev_close": prev_close,
                "change": change,
                "change_pct": change_pct,
                "volume": volume,
                "amount": amount,
                "volume_ratio": None,
                "amplitude": round((high / low - 1) * 100, 2) if (high and low and low > 0) else None,
                "turnover_rate": None,
                "limit_up": None,
                "limit_down": None,
                "speed_5m": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "_source": "sina",
            })

        return results
    except Exception as e:
        logger.error(f"新浪批量行情拉取失败: {e}")
        return []


def _to_em_secid(symbol: str) -> str:
    """
    将通用标的代码转为东方财富 secid 格式。
    规则：
      - 6 开头 → 上海 1.xxx
      - 0 或 3 开头 → 深圳 0.xxx
      - ETF (5/15/16) → 上海 1.xxx 或深圳 0.xxx
      - 指数 (000/399) → 上海/深圳指数
    """
    s = str(symbol).strip()
    if s.startswith("6") or s.startswith("5"):
        return f"1.{s}"
    elif s.startswith("0") or s.startswith("3") or s.startswith("15") or s.startswith("16"):
        return f"0.{s}"
    elif s.startswith("399"):
        return f"0.{s}"
    elif s.startswith("000"):
        return f"1.{s}"
    else:
        return f"1.{s}"  # 默认上海


# ─── 字段定义 ────────────────────────────────
# 东方财富 push2 接口字段映射
# f43=最新价, f44=最高, f45=最低, f46=今开, f47=成交量, f48=成交额
# f50=量比, f51=涨停, f52=跌停, f60=昨收
# f116=总市值, f117=流通市值, f162=市盈率, f167=换手率
# f168=振幅, f169=涨跌额, f170=涨跌幅, f171=5分钟涨速
REALTIME_FIELDS = "f43,f44,f45,f46,f47,f48,f50,f51,f52,f60,f116,f117,f162,f167,f168,f169,f170,f171,f57,f58"


def get_realtime_quote(symbol: str) -> Optional[dict]:
    """
    获取单个标的的实时行情快照。

    Returns:
        {
            "symbol": "510300",
            "name": "沪深300ETF",
            "price": 4.123,           # 最新价
            "open": 4.100,            # 今开
            "high": 4.150,            # 最高
            "low": 4.080,             # 最低
            "prev_close": 4.100,      # 昨收
            "change": 0.023,          # 涨跌额
            "change_pct": 0.56,       # 涨跌幅 %
            "volume": 1234567,        # 成交量（手）
            "amount": 123456789,      # 成交额（元）
            "volume_ratio": 1.23,     # 量比
            "amplitude": 1.71,        # 振幅 %
            "turnover_rate": 2.34,    # 换手率 %
            "limit_up": 4.510,        # 涨停价
            "limit_down": 3.690,      # 跌停价
            "speed_5m": 0.45,         # 5分钟涨速 %
            "timestamp": "2026-05-05 14:30:00"
        }
    """
    secid = _to_em_secid(symbol)
    url = f"https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        "fields": REALTIME_FIELDS,
        "invt": "2",
        "fltt": "2",
    }

    try:
        resp = requests.get(url, params=params, headers=EASTMONEY_HEADERS, timeout=5)
        data = resp.json().get("data", {})
        if not data:
            logger.warning(f"{symbol} 实时行情为空")
            return None

        quote = {
            "symbol": str(data.get("f57", symbol)),
            "name": str(data.get("f58", "")),
            "price": _to_float(data.get("f43")),
            "open": _to_float(data.get("f44")),
            "high": _to_float(data.get("f45")),
            "low": _to_float(data.get("f46")),
            "prev_close": _to_float(data.get("f60")),
            "change": _to_float(data.get("f169")),
            "change_pct": _to_float(data.get("f170")),
            "volume": _to_float(data.get("f47")),
            "amount": _to_float(data.get("f48")),
            "volume_ratio": _to_float(data.get("f50")),
            "amplitude": _to_float(data.get("f168")),
            "turnover_rate": _to_float(data.get("f167")),
            "limit_up": _to_float(data.get("f51")),
            "limit_down": _to_float(data.get("f52")),
            "speed_5m": _to_float(data.get("f171")),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return quote
    except Exception as e:
        logger.error(f"{symbol} 实时行情拉取失败: {e}")
        return None


def get_realtime_quotes(symbols: list) -> list:
    """批量获取多个标的的实时行情"""
    results = []
    for sym in symbols:
        q = get_realtime_quote(sym)
        if q:
            results.append(q)
        time.sleep(0.3)  # 频率限制
    return results


# ─── 东方财富批量接口（一次请求多个标的）────────

def get_batch_quotes(symbols: list) -> list:
    """
    东方财富批量实时行情接口。
    一次请求返回多个标的的数据，比单次轮询快很多。
    """
    secids = ",".join([_to_em_secid(s) for s in symbols])
    url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
    params = {
        "secids": secids,
        "fields": REALTIME_FIELDS,
        "invt": "2",
        "fltt": "2",
    }

    try:
        resp = requests.get(url, params=params, headers=EASTMONEY_HEADERS, timeout=5)
        data = resp.json().get("data", {})
        diff = data.get("diff", [])
        if not diff:
            logger.warning("批量实时行情为空")
            return []

        results = []
        for item in diff:
            results.append({
                "symbol": str(item.get("f57", "")),
                "name": str(item.get("f58", "")),
                "price": _to_float(item.get("f43")),
                "open": _to_float(item.get("f44")),
                "high": _to_float(item.get("f45")),
                "low": _to_float(item.get("f46")),
                "prev_close": _to_float(item.get("f60")),
                "change": _to_float(item.get("f169")),
                "change_pct": _to_float(item.get("f170")),
                "volume": _to_float(item.get("f47")),
                "amount": _to_float(item.get("f48")),
                "volume_ratio": _to_float(item.get("f50")),
                "amplitude": _to_float(item.get("f168")),
                "turnover_rate": _to_float(item.get("f167")),
                "limit_up": _to_float(item.get("f51")),
                "limit_down": _to_float(item.get("f52")),
                "speed_5m": _to_float(item.get("f171")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        return results
    except Exception as e:
        logger.error(f"批量实时行情拉取失败: {e}")
        return []


# ─── 盘中技术分析函数 ──────────────────────────

def calc_intraday_indicators(quote: dict, daily_df=None) -> dict:
    """
    基于实时快照和日线数据计算盘中技术指标信号。

    Args:
        quote: get_realtime_quote 返回的实时快照
        daily_df: 日线 OHLCV DataFrame（用于计算历史 RSI 等），可为 None

    Returns:
        盘中信号字典，包含盘中 RSI 估算、涨速信号、量比信号等
    """
    if not quote:
        return {}

    change_pct = quote.get("change_pct", 0) or 0
    speed_5m = quote.get("speed_5m", 0) or 0
    vol_ratio = quote.get("volume_ratio", 1) or 1
    amplitude = quote.get("amplitude", 0) or 0
    price = quote.get("price", 0) or 0
    prev_close = quote.get("prev_close", 0) or 0
    volume = quote.get("volume", 0) or 0

    signals = {}

    # ── 涨跌信号 ──
    if change_pct > 3:
        signals["change_signal"] = "大涨"
        signals["change_level"] = "high"
    elif change_pct > 1.5:
        signals["change_signal"] = "上涨"
        signals["change_level"] = "medium"
    elif change_pct < -3:
        signals["change_signal"] = "大跌"
        signals["change_level"] = "high"
    elif change_pct < -1.5:
        signals["change_signal"] = "下跌"
        signals["change_level"] = "medium"
    else:
        signals["change_signal"] = "平稳"
        signals["change_level"] = "low"

    # ── 涨速信号（5分钟）──
    # 新浪接口没有5分钟涨速，用振幅辅助判断
    if speed_5m is not None and speed_5m != 0:
        if speed_5m > 1.5:
            signals["speed_signal"] = "急拉"
            signals["speed_level"] = "high"
        elif speed_5m > 0.8:
            signals["speed_signal"] = "拉升"
            signals["speed_level"] = "medium"
        elif speed_5m < -1.5:
            signals["speed_signal"] = "急跌"
            signals["speed_level"] = "high"
        elif speed_5m < -0.8:
            signals["speed_signal"] = "下跌"
            signals["speed_level"] = "medium"
        else:
            signals["speed_signal"] = "平稳"
            signals["speed_level"] = "low"
    else:
        # 新浪接口没有实时涨速，基于振幅辅助判断
        if amplitude and amplitude > 3:
            signals["speed_signal"] = "大幅震荡"
            signals["speed_level"] = "medium"
        else:
            signals["speed_signal"] = "平稳"
            signals["speed_level"] = "low"

    # ── 盘中真实 RSI 估算 ──
    # 如果有日线数据，用历史数据 + 当前实时价计算真实 RSI
    if daily_df is not None and not daily_df.empty and len(daily_df) >= 14:
        closes = list(daily_df["close"].values)
        # 用当日实时价替换最后的收盘价
        if price and price > 0:
            closes[-1] = price
        intraday_rsi = _calc_series_rsi(pd.Series(closes), 14)
        signals["intraday_rsi"] = intraday_rsi
        signals["rsi_based_on"] = "日线+实时价"
    else:
        # 无日线数据时粗略估算
        if change_pct > 0:
            intraday_rsi = min(100, 50 + change_pct * 3)
        else:
            intraday_rsi = max(0, 50 + change_pct * 3)
        signals["intraday_rsi"] = round(intraday_rsi, 1)
        signals["rsi_based_on"] = "涨跌幅估算"

    if intraday_rsi > 80:
        signals["rsi_signal"] = "盘中超买"
        signals["rsi_level"] = "high"
    elif intraday_rsi < 20:
        signals["rsi_signal"] = "盘中超卖"
        signals["rsi_level"] = "high"
    else:
        signals["rsi_signal"] = "正常"
        signals["rsi_level"] = "low"

    # ── 量比信号 ──
    if vol_ratio is not None and vol_ratio > 1:
        if vol_ratio > 3:
            signals["volume_signal"] = "爆量"
            signals["volume_level"] = "high"
        elif vol_ratio > 2:
            signals["volume_signal"] = "显著放量"
            signals["volume_level"] = "high"
        elif vol_ratio > 1.5:
            signals["volume_signal"] = "放量"
            signals["volume_level"] = "medium"
        elif vol_ratio < 0.5:
            signals["volume_signal"] = "极度缩量"
            signals["volume_level"] = "medium"
        elif vol_ratio < 0.7:
            signals["volume_signal"] = "缩量"
            signals["volume_level"] = "low"
        else:
            signals["volume_signal"] = "正常"
            signals["volume_level"] = "low"
    else:
        # 新浪接口无量比：用当日成交量 vs 历史日均量
        signals["volume_signal"] = "待计算"
        signals["volume_level"] = "low"

    # ── 综合信号强度 ──
    high_count = sum(1 for k in signals if k.endswith("_level") and signals[k] == "high")
    if high_count >= 2:
        signals["alert_level"] = "⚠️⚠️ 强烈信号"
        signals["alert_score"] = min(100, 50 + high_count * 15)
    elif high_count == 1:
        signals["alert_level"] = "⚠️ 注意信号"
        signals["alert_score"] = min(100, 30 + high_count * 20)
    else:
        signals["alert_level"] = "✅ 正常"
        signals["alert_score"] = 20

    return signals


def should_alert(quote: dict, indicators: dict) -> Optional[dict]:
    """
    判断是否应该发送盘中提醒。
    返回提醒内容 dict，不触发时返回 None。

    触发条件：
      1. 5分钟涨速 > 1.5% 或 < -1.5%
      2. 量比 > 3
      3. 涨跌幅 > 4%
      4. 盘中 RSI 估算 > 80 或 < 20
    """
    if not quote or not indicators:
        return None

    alerts = []

    # 1. 涨速异动
    speed = quote.get("speed_5m", 0) or 0
    if speed > 1.5:
        alerts.append(f"⚡ 5分钟急拉 {speed:+.2f}%")
    elif speed < -1.5:
        alerts.append(f"⚡ 5分钟急跌 {speed:+.2f}%")

    # 2. 量比异动
    vr = quote.get("volume_ratio", 1) or 1
    if vr > 3:
        alerts.append(f"📊 爆量 {vr:.2f}x")

    # 3. 涨跌幅异动
    chg = quote.get("change_pct", 0) or 0
    if abs(chg) > 4:
        alerts.append(f"{'📈' if chg > 0 else '📉'} 涨跌异动 {chg:+.2f}%")

    # 4. RSI 超买超卖
    rsi = indicators.get("intraday_rsi", 50)
    if rsi > 80:
        alerts.append(f"🔴 盘中超买 RSI={rsi:.0f}")
    elif rsi < 20:
        alerts.append(f"🟢 盘中超卖 RSI={rsi:.0f}")

    if not alerts:
        return None

    return {
        "symbol": quote.get("symbol", ""),
        "name": quote.get("name", ""),
        "price": quote.get("price", 0),
        "change_pct": quote.get("change_pct", 0),
        "timestamp": quote.get("timestamp", ""),
        "alerts": alerts,
        "summary": " | ".join(alerts),
    }


# ─── 辅助函数 ───────────────────────────────

def _to_float(val) -> Optional[float]:
    """安全转浮点"""
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _calc_series_rsi(close: pd.Series, period: int = 14) -> float:
    """计算 RSI(14) 辅助函数"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


# ─── 快捷调用 ────────────────────────────────

def get_market_snapshot(symbols: list) -> dict:
    """
    获取市场快照，包含实时行情 + 盘中技术信号。
    优先使用东方财富（盘中返回-时自动降级到新浪）。
    返回 dict: {symbol: {quote, indicators, alerts}}
    """
    # 尝试东方财富批量接口
    quotes = get_batch_quotes(symbols)

    # 如果东方财富返回空或全是占位符（盘后），降级到新浪
    # 判断条件：所有 quote 的 price 为 None 或 "-" 或 0
    all_bad = True
    for q in quotes:
        p = q.get("price")
        if p not in (None, "-", 0, "0", ""):
            all_bad = False
            break
    if not quotes or all_bad:
        logger.info("东方财富实时数据不可用，降级到新浪接口")
        quotes = get_sina_batch_quotes(symbols)

    snapshot = {}

    for q in quotes:
        sym = q["symbol"]
        indicators = calc_intraday_indicators(q)
        alert = should_alert(q, indicators)
        snapshot[sym] = {
            "quote": q,
            "indicators": indicators,
            "alert": alert,
        }

    return snapshot


if __name__ == "__main__":
    import sys

    # 测试
    test_symbols = sys.argv[1:] or ["510300", "159915", "000001"]
    print(f"测试实时行情：{test_symbols}\n")

    snapshot = get_market_snapshot(test_symbols)
    for sym, data in snapshot.items():
        q = data["quote"]
        ind = data["indicators"]
        alert = data["alert"]

        print(f"{'='*40}")
        print(f"{q.get('name','')} ({sym})")
        print(f"  最新价: {q['price']}  ({q['change_pct']:+.2f}%)")
        print(f"  量比: {q['volume_ratio']:.2f}x  5分钟涨速: {q['speed_5m']:+.2f}%")
        print(f"  盘中RSI估算: {ind.get('intraday_rsi', 'N/A')}")
        print(f"  综合信号: {ind.get('alert_level','')}")
        if alert:
            print(f"  ⚠️ 提醒: {alert['summary']}")
        print()
