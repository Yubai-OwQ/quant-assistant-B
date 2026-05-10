"""
strategy/intraday_monitor.py
盘中实时监控模块 — 定时拉取实时行情，检测异动信号并推送

功能：
  - 盘中（9:30-15:00）按指定间隔拉取各标的一手实时行情
  - 基于实时数据计算盘中技术指标（RSI估算、涨速、量比等）
  - 异动触发时通过多渠道推送提醒
  - 生成盘中简报（可推送到微信/Telegram）

用法：
  python strategy/intraday_monitor.py                   # 启动监控
  python strategy/intraday_monitor.py --once            # 只跑一次并输出结果
  python strategy/intraday_monitor.py --interval 60     # 60秒间隔

依赖：
  pip install schedule (或集成到 scheduler.py 的 APScheduler)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "monitor.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("intraday_monitor")


# ─── 市场交易时间判断 ──────────────────────────

MARKET_OPEN = dtime(9, 30)    # 开盘
MORNING_CLOSE = dtime(11, 30)  # 午休
AFTERNOON_OPEN = dtime(13, 0)  # 下午开盘
MARKET_CLOSE = dtime(15, 0)    # 收盘

# 早盘集合竞价时段也拉（9:15-9:25）
PRE_OPEN = dtime(9, 15)
POST_CLOSE = dtime(15, 5)


def is_trading_time(now: datetime = None) -> bool:
    """判断当前是否为交易时间段（含集合竞价）"""
    if now is None:
        now = datetime.now()

    t = now.time()
    # 周末
    if now.weekday() >= 5:
        return False

    # 9:15-11:30 上午盘（含集合竞价）
    if PRE_OPEN <= t <= MORNING_CLOSE:
        return True
    # 13:00-15:05 下午盘（含收盘集合竞价）
    if AFTERNOON_OPEN <= t <= POST_CLOSE:
        return True

    return False


def should_poll(now: datetime = None) -> bool:
    """判断是否应该拉取数据（盘中 + 每5分钟最多拉一次守卫在外部）"""
    return is_trading_time(now)


# ─── 历史最高最低价跟踪（用于盘中突破判断）───────

class PriceTracker:
    """盘中价格记忆器，跟踪当日的最高最低和涨速"""

    def __init__(self):
        self.day_high = {}      # {symbol: price}
        self.day_low = {}       # {symbol: price}
        self.prev_price = {}    # {symbol: price} 上一轮价格
        self.trade_date = None

    def reset_if_new_day(self):
        """如果换了交易日，重置数据"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.trade_date != today:
            self.day_high.clear()
            self.day_low.clear()
            self.prev_price.clear()
            self.trade_date = today
            logger.info(f"交易日切换至 {today}，盘中数据已重置")

    def update(self, symbol: str, price: float):
        """更新盘中最高最低"""
        if symbol not in self.day_high or price > self.day_high[symbol]:
            self.day_high[symbol] = price
        if symbol not in self.day_low or price < self.day_low[symbol]:
            self.day_low[symbol] = price

    def get_intraday_range(self, symbol: str) -> dict:
        """获取盘中从最低到最高的涨幅"""
        low = self.day_low.get(symbol)
        high = self.day_high.get(symbol)
        if low and high and low > 0:
            return {
                "day_high": high,
                "day_low": low,
                "intraday_range_pct": round((high / low - 1) * 100, 2),
            }
        return {}

    def get_speed(self, symbol: str, price: float, interval_seconds: int) -> Optional[float]:
        """计算本次拉取相比上次的价格变化率（%/s 转换为 分钟涨速 %）"""
        prev = self.prev_price.get(symbol)
        if prev and prev > 0:
            rate = (price / prev - 1) * 100
            # 转换为每分钟涨速
            minute_rate = rate / (interval_seconds / 60)
            self.prev_price[symbol] = price
            return round(minute_rate, 2)
        else:
            self.prev_price[symbol] = price
            return None


tracker = PriceTracker()


# ─── 异动提醒推送（微信/本地）───────────────────

def send_alert(alert_data: dict):
    """
    发送异动提醒。
    目前打印到日志，可通过 cronjob 推送到微信。
    后续可以加企业微信/钉钉 webhook 等。
    """
    symbol = alert_data.get("symbol", "")
    name = alert_data.get("name", "")
    summary = alert_data.get("summary", "")
    price = alert_data.get("price", 0)
    change_pct = alert_data.get("change_pct", 0)

    msg = (
        f"⚡ 盘中异动\n"
        f"{name} ({symbol})\n"
        f"现价: {price}  ({change_pct:+.2f}%)\n"
        f"{summary}\n"
        f"时间: {alert_data.get('timestamp', '')}"
    )

    logger.warning(f"🔔 异动提醒:\n{msg}")

    # 保存到文件，供外部 cronjob 读取
    alert_path = ROOT_DIR / "alerts" / f"alert_{datetime.now().strftime('%H%M%S')}.json"
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alert_path, "w", encoding="utf-8") as f:
        json.dump(alert_data, f, ensure_ascii=False, indent=2)

    return msg


# ─── 止损监控 ────────────────────────────────
def check_stop_loss(symbol: str, price: float, name: str = "") -> Optional[dict]:
    """
    检查 config.yaml 中配置的止损监控。
    如果当前亏损达到或超过阈值，返回提醒 dict。
    """
    try:
        import yaml
        cfg_path = ROOT_DIR / "config.yaml"
        if not cfg_path.exists():
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sl_watch = cfg.get("monitor", {}).get("stop_loss_watch", {})
        if symbol not in sl_watch:
            return None

        watch = sl_watch[symbol]
        buy_price = watch.get("buy_price", 0)
        stop_loss_pct = watch.get("stop_loss_pct", 5)
        remind_at_pct = watch.get("remind_at_pct", 4)

        if not buy_price or buy_price <= 0:
            return None

        loss_pct = (price / buy_price - 1) * 100  # 负值 = 亏损

        # 达到止损位
        if loss_pct <= -stop_loss_pct:
            return {
                "type": "stop_loss",
                "symbol": symbol,
                "name": name or symbol,
                "price": price,
                "buy_price": buy_price,
                "loss_pct": round(loss_pct, 2),
                "action": "❗ **触及止损线！**",
                "advice": "建议立即卖出",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }

        # 接近止损（提醒）
        if loss_pct <= -remind_at_pct:
            remaining = abs(loss_pct + stop_loss_pct)
            return {
                "type": "stop_loss_warning",
                "symbol": symbol,
                "name": name or symbol,
                "price": price,
                "buy_price": buy_price,
                "loss_pct": round(loss_pct, 2),
                "remaining_to_stop": round(remaining, 2),
                "action": "⚠️ **接近止损线！**",
                "advice": f"距止损仅差 {remaining:.2f}%，密切关注",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
    except Exception as e:
        logger.debug(f"止损监控检查失败: {e}")

    return None


def send_stop_loss_alert(alert: dict):
    """发送止损提醒"""
    msg = (
        f"{alert['action']}\n"
        f"{alert['name']} ({alert['symbol']})\n"
        f"买入价: {alert['buy_price']}\n"
        f"现价: {alert['price']}  (亏损 {alert['loss_pct']:.2f}%)\n"
        f"{alert['advice']}\n"
        f"时间: {alert['timestamp']}"
    )
    logger.warning(f"🔔 {msg}")

    alert_path = ROOT_DIR / "alerts" / f"stop_loss_{alert['symbol']}_{datetime.now().strftime('%H%M%S')}.json"
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alert_path, "w", encoding="utf-8") as f:
        json.dump(alert, f, ensure_ascii=False, indent=2)

    return msg


# ─── 盘中简报生成 ──────────────────────────────

def generate_brief(snapshot: dict) -> str:
    """
    生成当前盘中行情简报（Markdown格式）：
    - 各标的实时行情
    - 异动信号汇总
    - 综合提示

    Returns: Markdown 文本
    """
    lines = [f"⏰ 盘中简报 ({datetime.now().strftime('%H:%M:%S')})", ""]

    for sym, data in sorted(snapshot.items()):
        q = data.get("quote", {})
        ind = data.get("indicators", {})

        name = q.get("name", sym)
        price = q.get("price", "-")
        chg = q.get("change_pct", 0)
        chg_str = f"{chg:+.2f}%" if chg is not None else "-"
        vr = q.get("volume_ratio", "-")
        speed = q.get("speed_5m", "-")

        # 方向图标
        if chg and chg > 0:
            icon = "🟢"
        elif chg and chg < 0:
            icon = "🔴"
        else:
            icon = "⚪"

        lines.append(f"{icon} **{name}**  {price}  ({chg_str})")
        lines.append(f"  量比 {vr}x | 5分钟涨速 {speed:+.2f}%" if isinstance(speed, (int, float)) else "")

        # 信号
        signals = []
        alert_level = ind.get("alert_level", "")
        if "强烈" in alert_level:
            signals.append(alert_level)
        elif "注意" in alert_level:
            signals.append(alert_level)

        rsi_signal = ind.get("rsi_signal", "")
        vol_signal = ind.get("volume_signal", "")
        speed_signal = ind.get("speed_signal", "")

        if rsi_signal not in ("正常", ""):
            signals.append(f"RSI({rsi_signal})")
        if vol_signal not in ("正常", ""):
            signals.append(f"量({vol_signal})")
        if speed_signal not in ("平稳", ""):
            signals.append(f"速({speed_signal})")

        if signals:
            lines.append(f"  ⚠️ {' '.join(signals)}")

        lines.append("")

    # 提醒汇总
    alerts = []
    for sym, data in sorted(snapshot.items()):
        alert = data.get("alert")
        if alert:
            alerts.append(alert["summary"])

    if alerts:
        lines.append("---")
        lines.append("**⚠️ 异动提醒**")
        for a in alerts:
            lines.append(f"  {a}")

    return "\n".join(lines)


# ─── 主轮询逻辑 ──────────────────────────────

def poll_once(symbols: list = None) -> dict:
    """
    执行一次盘中行情拉取 + 异动检测。

    Args:
        symbols: 标的列表，None 则从 config.yaml 读取

    Returns:
        (snapshot dict, alerts list)
    """
    if symbols is None:
        import yaml
        cfg_path = ROOT_DIR / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            symbols = [str(s) for s in cfg.get("strategy", {}).get("symbols", ["510300", "159915"])]
        else:
            symbols = ["510300", "159915"]

    # 先读日线数据（用于盘中 RSI 计算）
    from data.fetcher import get_ohlcv
    daily_data = {}
    for s in symbols:
        try:
            df = get_ohlcv(s, days=120)
            if df is not None and not df.empty:
                daily_data[s] = df
        except Exception:
            pass

    from data.realtime import get_market_snapshot
    from data.realtime import calc_intraday_indicators, should_alert as rt_should_alert
    from data.realtime import get_sina_batch_quotes, get_batch_quotes

    # 盘中直接走新浪接口（东方财富盘后和盘中都返回占位符，不稳定）
    quotes = get_sina_batch_quotes(symbols)
    # 过滤掉无效数据
    valid_quotes = []
    for q in quotes:
        price = q.get("price")
        name = q.get("name", "")
        if price and isinstance(price, (int, float)) and price > 0 and name and name != "-":
            valid_quotes.append(q)
    quotes = valid_quotes

    tracker.reset_if_new_day()
    alerts_triggered = []
    snapshot = {}

    for q in quotes:
        sym = q.get("symbol", "")
        if not sym:
            continue

        # 传入日线数据算盘中 RSI
        df = daily_data.get(sym)
        indicators = calc_intraday_indicators(q, daily_df=df)
        alert = rt_should_alert(q, indicators)

        snapshot[sym] = {
            "quote": q,
            "indicators": indicators,
            "alert": alert,
        }

        price = q.get("price", 0)
        if price and isinstance(price, (int, float)) and price > 0:
            tracker.update(sym, price)

        if alert:
            alerts_triggered.append(alert)
            send_alert(alert)
            _save_alert_to_db(alert)

        # 止损监控
        name = q.get("name", "")
        sl_alert = check_stop_loss(sym, price, name)
        if sl_alert:
            alerts_triggered.append(sl_alert)
            send_stop_loss_alert(sl_alert)
            _save_alert_to_db(sl_alert)

    return snapshot, alerts_triggered


def _save_alert_to_db(alert: dict):
    """保存异动记录到数据库"""
    import sqlite3
    db_path = ROOT_DIR / "db" / "quant.db"
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intraday_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                name TEXT,
                price REAL,
                change_pct REAL,
                summary TEXT,
                timestamp TEXT,
                created_at TEXT
            )
        """)
        summary = alert.get("summary") or alert.get("advice") or alert.get("action") or str(alert.get("type", ""))
        conn.execute(
            "INSERT INTO intraday_alerts (symbol, name, price, change_pct, summary, timestamp, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                alert.get("symbol", ""),
                alert.get("name", ""),
                alert.get("price", 0),
                alert.get("change_pct", 0),
                summary,
                alert.get("timestamp", ""),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"写入异动到数据库失败: {e}")


def poll_loop(interval: int = 60, symbols: list = None):
    """
    盘中循环拉取，每次轮询间隔 interval 秒。
    只在交易时间执行，非交易时间自动暂停。
    """
    logger.info(f"盘中监控启动，轮询间隔 {interval} 秒，标的: {symbols or 'config.yaml'}")

    iteration = 0
    while True:
        now = datetime.now()
        if not should_poll(now):
            next_check = "09:15" if now.time() < PRE_OPEN else "第二天 09:15"
            logger.info(f"非交易时间，暂停监控（下一轮检查: {next_check}）")
            # 非交易时间每5分钟检查一次是否开盘
            time.sleep(300)
            continue

        iteration += 1
        now_str = now.strftime("%H:%M:%S")
        logger.info(f"[{now_str}] 第 {iteration} 轮盘中轮询...")

        try:
            snapshot, alerts = poll_once(symbols)

            # 每5轮输出一次简报
            if iteration % 5 == 0:
                brief = generate_brief(snapshot)
                logger.info(f"📋 盘中简报:\n{brief}")

        except Exception as e:
            logger.error(f"轮询异常: {e}", exc_info=True)

        time.sleep(interval)


# ─── 命令行入口 ──────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="盘中实时监控")
    parser.add_argument("--once", action="store_true", help="只执行一次并输出结果")
    parser.add_argument("--interval", type=int, default=60, help="轮询间隔（秒，默认60）")
    parser.add_argument("--symbols", nargs="+", help="标的列表，不填则从 config.yaml 读取")
    args = parser.parse_args()

    if args.once:
        snapshot, alerts = poll_once(args.symbols)
        brief = generate_brief(snapshot)
        print(brief)
        if alerts:
            print(f"\n⚠️ 本次触发 {len(alerts)} 条异动")
    else:
        poll_loop(interval=args.interval, symbols=args.symbols)


if __name__ == "__main__":
    main()
