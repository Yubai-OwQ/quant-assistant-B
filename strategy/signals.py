"""
strategy/signals.py
技术因子计算 + 信号生成模块

功能：
  - 计算 MACD / RSI / 布林带 / KDJ / OBV / 量比 等技术因子
  - 将多因子加权合成 0-100 综合评分
  - 调用 DeepSeek AI 输出最终交易方向、仓位、止损建议
  - 将信号写入 SQLite signals 表
  - 支持手动触发和 APScheduler 定时调用

数据库表（自动创建）：
  signals(id, symbol, signal_date, direction, confidence,
          suggested_position, stop_loss, reasoning, composite_score,
          actual_return, is_correct)

依赖：
  pip install pandas pandas-ta openai pyyaml

用法：
  python strategy/signals.py              # 对所有标的生成今日信号
  python strategy/signals.py --symbol 000300
  python strategy/signals.py --dry-run    # 只打印不写库
"""

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────
# 路径 & 日志
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
        logging.FileHandler(str(LOG_DIR / "signals.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("signals")


# ─────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────
def load_config() -> dict:
    try:
        import yaml
        cfg_path = ROOT_DIR / "config.yaml"
        if not cfg_path.exists():
            return _default_config()
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or _default_config()
    except Exception as e:
        logger.warning(f"读取配置失败：{e}，使用默认配置")
        return _default_config()


def _default_config() -> dict:
    return {
        "strategy": {"style": "balanced", "symbols": ["000300", "510050"]},
        "risk":     {"max_position": 0.6, "stop_loss": 0.05, "take_profit": 0.12},
        "signals":  {
            "rsi_overbought": 70, "rsi_oversold": 30,
            "macd_weight": 0.4, "rsi_weight": 0.3, "volume_weight": 0.3,
        },
    }


# ─────────────────────────────────────────────
# 数据库初始化
# ─────────────────────────────────────────────
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()


def _save_signal(signal: dict) -> bool:
    """
    写入单条信号，同一标的同一天已存在则覆盖（UPDATE）。
    返回 True 表示成功。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
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
            signal["symbol"],
            signal["signal_date"],
            signal["direction"],
            signal["confidence"],
            signal["suggested_position"],
            signal["stop_loss"],
            signal["reasoning"],
            signal["composite_score"],
        ))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        logger.error(f"写入信号失败：{e}")
        return False


# ─────────────────────────────────────────────
# 技术因子计算
# ─────────────────────────────────────────────
def _calc_macd(close: pd.Series,
               fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    计算 MACD，返回最新值。
    使用纯 pandas 实现，不依赖 TA-Lib。
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    dif        = ema_fast - ema_slow
    dea        = dif.ewm(span=signal,   adjust=False).mean()
    hist       = (dif - dea) * 2

    return {
        "dif":  round(float(dif.iloc[-1]),  6),
        "dea":  round(float(dea.iloc[-1]),  6),
        "hist": round(float(hist.iloc[-1]), 6),
        # 金叉：dif 上穿 dea（昨日 dif < dea，今日 dif > dea）
        "golden_cross": bool(
            len(dif) >= 2 and dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]
        ),
        "dead_cross": bool(
            len(dif) >= 2 and dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]
        ),
    }


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    """计算 RSI(14)"""
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, float("inf"))
    rsi    = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def _calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """计算布林带，返回上中下轨及价格在带内位置（0=下轨，1=上轨）"""
    ma    = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std

    last_close = float(close.iloc[-1])
    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])
    last_ma    = float(ma.iloc[-1])
    band_width = last_upper - last_lower

    position = (last_close - last_lower) / band_width if band_width > 0 else 0.5

    return {
        "upper":    round(last_upper, 4),
        "middle":   round(last_ma,    4),
        "lower":    round(last_lower, 4),
        "position": round(position,   4),   # 0~1，> 0.8 超买，< 0.2 超卖
        "width_pct": round(band_width / last_ma * 100, 2) if last_ma > 0 else 0,
    }


def _calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
              n: int = 9) -> dict:
    """计算 KDJ"""
    low_n  = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv    = (close - low_n) / (high_n - low_n).replace(0, 1) * 100

    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2,   adjust=False).mean()
    j = 3 * k - 2 * d

    return {
        "k": round(float(k.iloc[-1]), 2),
        "d": round(float(d.iloc[-1]), 2),
        "j": round(float(j.iloc[-1]), 2),
    }


def _calc_volume_ratio(volume: pd.Series, period: int = 20) -> float:
    """计算量比：最新成交量 / 近 N 日平均成交量"""
    if len(volume) < period:
        return 1.0
    avg_vol = float(volume.iloc[-period:-1].mean())
    if avg_vol <= 0:
        return 1.0
    return round(float(volume.iloc[-1]) / avg_vol, 2)


def _calc_obv(close: pd.Series, volume: pd.Series) -> dict:
    """
    计算 OBV（能量潮），返回最新值和趋势方向。
    OBV 上升表示资金流入（看涨），下降表示资金流出（看跌）。
    """
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv       = (volume * direction).cumsum()
    obv_ma5   = obv.rolling(5).mean()

    trend = "up" if float(obv.iloc[-1]) > float(obv_ma5.iloc[-1]) else "down"
    return {
        "obv":      round(float(obv.iloc[-1]),    2),
        "obv_ma5":  round(float(obv_ma5.iloc[-1]), 2),
        "trend":    trend,
    }


# ─────────────────────────────────────────────
# 综合评分（0-100）
# ─────────────────────────────────────────────
def _composite_score(
    macd:   dict,
    rsi:    float,
    boll:   dict,
    vol_ratio: float,
    obv:    dict,
    config: dict,
) -> float:
    """
    将各因子信号归一化后加权合成 0-100 分。
    > 60 倾向做多，< 40 倾向做空，40-60 观望。
    """
    sig_cfg = config.get("signals", {})
    w_macd  = sig_cfg.get("macd_weight",   0.4)
    w_rsi   = sig_cfg.get("rsi_weight",    0.3)
    w_vol   = sig_cfg.get("volume_weight", 0.3)

    rsi_ob = sig_cfg.get("rsi_overbought", 70)
    rsi_os = sig_cfg.get("rsi_oversold",   30)

    # ── MACD 分数（0-100）──────────────────
    if macd["hist"] > 0:
        # 柱线为正：看涨
        macd_score = 60 + min(40, abs(macd["hist"]) * 2000)
    else:
        macd_score = 40 - min(40, abs(macd["hist"]) * 2000)

    if macd["golden_cross"]:
        macd_score = min(100, macd_score + 15)
    if macd["dead_cross"]:
        macd_score = max(0,   macd_score - 15)

    # ── RSI 分数（0-100）───────────────────
    if rsi <= rsi_os:
        # 超卖区，看涨信号
        rsi_score = 75 + (rsi_os - rsi) / rsi_os * 25
    elif rsi >= rsi_ob:
        # 超买区，看跌信号
        rsi_score = 25 - (rsi - rsi_ob) / (100 - rsi_ob) * 25
    else:
        # 中性区，线性映射到 30-70
        rsi_score = 30 + (rsi - rsi_os) / (rsi_ob - rsi_os) * 40

    # ── 量价分数（0-100）──────────────────
    # 量比 > 1.5 且 OBV 趋势向上 → 强烈看涨
    if vol_ratio > 1.5 and obv["trend"] == "up":
        vol_score = 75
    elif vol_ratio > 1.2 and obv["trend"] == "up":
        vol_score = 65
    elif vol_ratio < 0.7:
        vol_score = 35   # 缩量
    elif obv["trend"] == "down":
        vol_score = 40
    else:
        vol_score = 50

    # ── 布林带补正（±5分）─────────────────
    boll_adj = 0.0
    if boll["position"] < 0.1:
        boll_adj = +5   # 触碰下轨，超卖补正
    elif boll["position"] > 0.9:
        boll_adj = -5   # 触碰上轨，超买补正

    score = w_macd * macd_score + w_rsi * rsi_score + w_vol * vol_score + boll_adj
    return round(max(0.0, min(100.0, score)), 1)


# ─────────────────────────────────────────────
# DeepSeek AI 决策
# ─────────────────────────────────────────────
def _build_signal_prompt(symbol: str, market: dict, style: str, config: dict) -> str:
    r = config.get("risk", {})
    style_desc = {
        "aggressive":   "激进——高仓位，接受较大波动，止损可宽至8%",
        "balanced":     "均衡——中等仓位，收益与风险平衡",
        "conservative": "保守——低仓位，严格止损，宁可错过不可错拿",
    }.get(style, "均衡")

    return f"""你是专业量化交易分析师，请根据以下技术面数据给出今日交易建议。

## 标的：{symbol}  |  策略风格：{style_desc}

## 技术指标
- 当前价格：{market['close']:.3f}
- RSI(14)：{market['rsi']:.1f}
- MACD 柱（hist）：{market['macd_hist']:.6f}
- MACD 金叉：{market['macd_golden']}  死叉：{market['macd_dead']}
- 布林带位置：{market['bb_position']:.2f}（0=下轨，1=上轨）
- 布林带宽度：{market['bb_width_pct']:.1f}%
- 量比：{market['vol_ratio']:.2f}（>1 放量，<1 缩量）
- OBV 趋势：{market['obv_trend']}
- KDJ K={market['kdj_k']:.1f} D={market['kdj_d']:.1f} J={market['kdj_j']:.1f}
- 综合因子评分：{market['composite_score']:.1f}/100

## 风险参数参考
- 最大仓位上限：{r.get('max_position', 0.6) * 100:.0f}%
- 默认止损：{r.get('stop_loss', 0.05) * 100:.0f}%

## 输出要求
只输出如下 JSON，不要其他任何文字：
{{
  "direction": "buy 或 sell 或 hold",
  "confidence": 0到100的整数,
  "suggested_position": 0.0到{r.get('max_position', 0.6)}之间的小数,
  "stop_loss": 0.01到0.10之间的小数,
  "reasoning": "50字以内的分析理由"
}}
"""


def _call_deepseek(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """调用 DeepSeek API，返回解析后的 JSON，失败返回 None"""
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败（第{attempt}次）：{e}")
        except Exception as e:
            logger.warning(f"DeepSeek 调用失败（第{attempt}次）：{e}")
            import time; import time as t; t.sleep(2 * attempt)

    return None


def _fallback_signal(composite_score: float, config: dict) -> dict:
    """
    AI 调用失败时的规则兜底：纯基于综合评分输出信号。
    """
    r = config.get("risk", {})
    if composite_score >= 65:
        direction  = "buy"
        confidence = int(composite_score)
        position   = r.get("max_position", 0.6) * (composite_score - 60) / 40
    elif composite_score <= 35:
        direction  = "sell"
        confidence = int(100 - composite_score)
        position   = r.get("max_position", 0.6) * (40 - composite_score) / 40
    else:
        direction  = "hold"
        confidence = 50
        position   = 0.0

    return {
        "direction":          direction,
        "confidence":         min(100, max(0, confidence)),
        "suggested_position": round(min(r.get("max_position", 0.6), max(0.0, position)), 2),
        "stop_loss":          r.get("stop_loss", 0.05),
        "reasoning":          f"AI不可用，规则兜底：综合评分{composite_score:.0f}",
    }


# ─────────────────────────────────────────────
# 单标的信号生成
# ─────────────────────────────────────────────
def generate_signal(symbol: str, dry_run: bool = False) -> Optional[dict]:
    """
    对单个标的生成今日信号。

    流程：
      1. 从数据库读取 OHLCV
      2. 计算技术因子
      3. 调用 DeepSeek 输出方向/仓位/止损
      4. 写入 signals 表

    Args:
        symbol:  标的代码
        dry_run: True 时只打印，不写数据库

    Returns:
        信号 dict，失败返回 None
    """
    # ── 1. 读取行情 ─────────────────────────
    from data.fetcher import get_ohlcv
    df = get_ohlcv(symbol, days=120)

    if df.empty or len(df) < 30:
        logger.warning(f"{symbol} 数据不足（{len(df)} 条），需至少 30 条。"
                       f"请先运行 data/fetcher.py 拉取数据。")
        return None

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"].fillna(0)

    # ── 2. 计算技术因子 ──────────────────────
    config = load_config()
    macd   = _calc_macd(close)
    rsi    = _calc_rsi(close)
    boll   = _calc_bollinger(close)
    kdj    = _calc_kdj(high, low, close)
    vol_r  = _calc_volume_ratio(volume)
    obv    = _calc_obv(close, volume)
    score  = _composite_score(macd, rsi, boll, vol_r, obv, config)

    market = {
        "close":           round(float(close.iloc[-1]), 4),
        "rsi":             rsi,
        "macd_hist":       macd["hist"],
        "macd_golden":     macd["golden_cross"],
        "macd_dead":       macd["dead_cross"],
        "bb_position":     boll["position"],
        "bb_width_pct":    boll["width_pct"],
        "vol_ratio":       vol_r,
        "obv_trend":       obv["trend"],
        "kdj_k":           kdj["k"],
        "kdj_d":           kdj["d"],
        "kdj_j":           kdj["j"],
        "composite_score": score,
    }

    logger.info(
        f"{symbol} 因子：RSI={rsi:.1f} MACD柱={macd['hist']:.5f} "
        f"BB位置={boll['position']:.2f} 量比={vol_r:.2f} 评分={score:.1f}"
    )

    # ── 3. AI 决策 ──────────────────────────
    style  = config.get("strategy", {}).get("style", "balanced")
    prompt = _build_signal_prompt(symbol, market, style, config)

    try:
        ai_result = _call_deepseek(prompt)
    except EnvironmentError as e:
        logger.warning(f"API Key 未设置，使用规则兜底：{e}")
        ai_result = None

    if ai_result is None:
        logger.warning(f"{symbol} DeepSeek 调用失败，使用规则兜底")
        ai_result = _fallback_signal(score, config)

    # ── 4. 组装信号 ──────────────────────────
    today  = datetime.now().strftime("%Y-%m-%d")
    signal = {
        "symbol":            symbol,
        "signal_date":       today,
        "direction":         ai_result.get("direction", "hold"),
        "confidence":        int(ai_result.get("confidence", 50)),
        "suggested_position": float(ai_result.get("suggested_position", 0.0)),
        "stop_loss":         float(ai_result.get("stop_loss", config.get("risk", {}).get("stop_loss", 0.05))),
        "reasoning":         str(ai_result.get("reasoning", "")),
        "composite_score":   score,
    }

    direction_icon = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(signal["direction"], "⚪")
    logger.info(
        f"{symbol} 信号：{direction_icon} {signal['direction'].upper()} "
        f"置信度={signal['confidence']} 仓位={signal['suggested_position']*100:.0f}% "
        f"止损={signal['stop_loss']*100:.1f}%"
    )
    logger.info(f"{symbol} 理由：{signal['reasoning']}")

    # ── 5. 写入数据库 ─────────────────────────
    if not dry_run:
        init_db()
        ok = _save_signal(signal)
        if ok:
            logger.info(f"{symbol} 信号已写入数据库")
        else:
            logger.error(f"{symbol} 信号写入失败")

    return signal


# ─────────────────────────────────────────────
# 批量生成（读取 config.yaml）
# ─────────────────────────────────────────────
def run_all(dry_run: bool = False) -> list:
    """
    对 config.yaml 中所有标的生成今日信号。

    Returns:
        成功生成的信号列表
    """
    import time
    init_db()
    config  = load_config()
    symbols = config.get("strategy", {}).get("symbols", ["000300", "510050"])
    results = []

    logger.info(f"开始生成信号，共 {len(symbols)} 个标的")

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] 处理 {symbol}")
        signal = generate_signal(str(symbol), dry_run=dry_run)
        if signal:
            results.append(signal)
        if i < len(symbols):
            time.sleep(1.5)   # API 调用间隔

    logger.info(f"信号生成完成：{len(results)}/{len(symbols)} 成功")
    return results


# ─────────────────────────────────────────────
# APScheduler 注册函数（供 scheduler.py 调用）
# ─────────────────────────────────────────────
def scheduled_generate() -> None:
    """
    每日 09:15 自动触发，在 scheduler.py 中注册：

        from strategy.signals import scheduled_generate
        scheduler.add_job(scheduled_generate, 'cron', hour=9, minute=15)
    """
    try:
        signals = run_all(dry_run=False)
        logger.info(f"定时信号生成完成，共 {len(signals)} 条")
    except Exception as e:
        logger.error(f"定时信号生成异常：{e}", exc_info=True)


# ─────────────────────────────────────────────
# scheduler.py 适配函数
# ─────────────────────────────────────────────
def calculate_all_signals(df: pd.DataFrame, config: dict) -> dict:
    """
    scheduler.py 兼容接口。
    对传入的 DataFrame 计算全部技术因子，返回字典。
    返回格式兼容 ai_engine.analyze_market()。
    """
    if df.empty or len(df) < 30:
        return {
            "composite": 0, "macd": 0, "rsi_value": 50, "bollinger": 0,
            "volume": 0, "trend": 0, "price_change_pct": 0,
        }

    close  = df["close"]
    high   = df.get("high", close)
    low    = df.get("low", close)
    volume = df.get("volume", pd.Series([0] * len(close))).fillna(0)

    macd   = _calc_macd(close)
    rsi    = _calc_rsi(close)
    boll   = _calc_bollinger(close)
    vol_r  = _calc_volume_ratio(volume)
    obv    = _calc_obv(close, volume)
    score  = _composite_score(macd, rsi, boll, vol_r, obv, config)

    # 趋势：短期均线比较
    ma5  = float(close.tail(5).mean())
    ma20 = float(close.tail(20).mean())
    trend = round((ma5 / ma20 - 1) * 100, 4) if ma20 > 0 else 0

    pct = float(close.pct_change().iloc[-1] * 100) if len(close) >= 2 else 0

    return {
        "composite":        round(score / 100, 4),      # 归一化到 -1~1 风格
        "macd":             round(macd["hist"], 4),
        "rsi_value":        rsi,
        "bollinger":        round(boll["position"] - 0.5, 4),
        "volume":           round(vol_r - 1.0, 4),
        "trend":            trend,
        "price_change_pct": round(pct, 2),
    }


def run_backtest(df: pd.DataFrame, config: dict, window: int = 30) -> dict:
    """
    scheduler.py 兼容接口。
    基于技术信号在历史数据上做简单回测。
    Returns: {sharpe, max_drawdown, win_rate, total_return, n_trades}
    """
    if df.empty or len(df) < window + 30:
        return {"sharpe": 0, "max_drawdown": 0, "win_rate": 0, "total_return": 0, "n_trades": 0}

    close  = df["close"].values
    dates  = df["date"].values

    r_cfg  = config.get("risk", {})
    sig_cfg = config.get("signals", {})

    # 在历史数据上逐日滑动计算信号
    positions = []
    lookback = 60
    daily_rets = []

    for i in range(lookback, len(close) - 1):
        seg_close  = pd.Series(close[i - lookback:i + 1])
        seg_high   = pd.Series(df["high"].values[i - lookback:i + 1])
        seg_low    = pd.Series(df["low"].values[i - lookback:i + 1])
        seg_volume = pd.Series(df["volume"].values[i - lookback:i + 1])

        macd  = _calc_macd(seg_close)
        rsi   = _calc_rsi(seg_close)
        boll  = _calc_bollinger(seg_close)
        vol_r = _calc_volume_ratio(seg_volume)
        obv   = _calc_obv(seg_close, seg_volume)
        score = _composite_score(macd, rsi, boll, vol_r, obv, config)

        if score >= 63:
            pos = min(r_cfg.get("max_position", 0.6), 0.3 + (score - 60) / 40 * 0.4)
        elif score <= 37:
            pos = -min(r_cfg.get("max_position", 0.6), 0.3 + (40 - score) / 40 * 0.4)
        else:
            pos = 0.0

        # 次日收益
        next_ret = (close[i + 1] - close[i]) / close[i]
        daily_rets.append(pos * next_ret)
        positions.append((dates[i], pos))

    if not daily_rets:
        return {"sharpe": 0, "max_drawdown": 0, "win_rate": 0, "total_return": 0, "n_trades": 0}

    rets = pd.Series(daily_rets)
    total_ret = float((1 + rets).prod() - 1)
    sharpe    = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0
    cum       = (1 + rets).cumprod()
    peak      = cum.cummax()
    dd        = ((cum - peak) / peak).min()
    max_dd    = float(abs(dd))
    win_rate  = float((rets > 0).sum() / len(rets))
    n_trades  = sum(1 for i in range(1, len(positions))
                    if positions[i][1] != 0 and positions[i - 1][1] == 0)

    return {
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_dd, 4),
        "win_rate":      round(win_rate, 4),
        "total_return":  round(total_ret, 4),
        "n_trades":      n_trades,
    }


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股技术信号生成工具")
    parser.add_argument("--symbol",  type=str, default=None,
                        help="指定标的代码（不填则处理 config.yaml 中所有标的）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印信号，不写入数据库")
    args = parser.parse_args()

    if args.symbol:
        signal = generate_signal(args.symbol, dry_run=args.dry_run)
        if signal:
            print("\n── 信号结果 ──────────────────────────────")
            icon = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(signal["direction"], "⚪")
            print(f"  标的：{signal['symbol']}   日期：{signal['signal_date']}")
            print(f"  方向：{icon} {signal['direction'].upper()}")
            print(f"  置信度：{signal['confidence']}   综合评分：{signal['composite_score']}")
            print(f"  建议仓位：{signal['suggested_position']*100:.0f}%")
            print(f"  止损：{signal['stop_loss']*100:.1f}%")
            print(f"  理由：{signal['reasoning']}")
            if args.dry_run:
                print("\n  [DRY RUN] 未写入数据库")
        else:
            print(f"❌ {args.symbol} 信号生成失败，请查看 logs/signals.log")
    else:
        signals = run_all(dry_run=args.dry_run)
        print(f"\n── 批量信号结果 ─────────────────────────────")
        for s in signals:
            icon = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(s["direction"], "⚪")
            print(f"  {icon} {s['symbol']:<12} {s['direction'].upper():<6} "
                  f"置信度={s['confidence']:<4} 仓位={s['suggested_position']*100:.0f}%  "
                  f"{s['reasoning'][:30]}")
        if args.dry_run:
            print("\n  [DRY RUN] 未写入数据库")
