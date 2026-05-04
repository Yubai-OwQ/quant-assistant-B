"""
app.py
量化助手 Gradio 对话界面
 
功能：
  - 自然语言对话控制策略参数（切换风格、调整止损、查询绩效等）
  - 实时展示当前策略信号和仓位建议
  - 回测结果可视化（K线 + 收益曲线 + 回撤）
  - 手动触发数据拉取、信号生成、调优
 
启动：
  python app.py
访问：
  http://localhost:7860
 
依赖：
  pip install gradio openai pyyaml plotly pandas
"""
 
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
 
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yaml
from openai import OpenAI
from pathlib import Path as _Path
 
# ─────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("app")
 
# ─────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
DB_PATH = ROOT_DIR / "db" / "quant.db"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
 
# ─────────────────────────────────────────────
# DeepSeek 客户端
# ─────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
 
 
# ─────────────────────────────────────────────
# 配置读写
# ─────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return _default_config()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or _default_config()
 
 
def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    logger.info("config.yaml 已更新")
 
 
def _default_config() -> dict:
    return {
        "strategy": {
            "style": "balanced",
            "market": "A股",
            "symbols": ["000300", "510050"],
        },
        "risk": {
            "max_position": 0.6,
            "stop_loss": 0.05,
            "take_profit": 0.12,
        },
        "signals": {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_weight": 0.4,
            "rsi_weight": 0.3,
            "volume_weight": 0.3,
        },
        "schedule": {
            "data_fetch": "09:00",
            "signal_gen": "09:15",
            "optimizer": "22:00",
        },
    }
 
 
def config_to_markdown(config: dict) -> str:
    """将配置格式化为可读的 Markdown 表格"""
    s = config.get("strategy", {})
    r = config.get("risk", {})
    sig = config.get("signals", {})
 
    style_map = {
        "aggressive": "🔴 激进",
        "balanced": "🟡 均衡",
        "conservative": "🟢 保守",
    }
    style_label = style_map.get(s.get("style", "balanced"), s.get("style", "-"))
 
    lines = [
        "### 当前策略配置",
        "",
        f"**风格**：{style_label}　　**市场**：{s.get('market', '-')}",
        f"**标的**：{', '.join(s.get('symbols', []))}",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 最大仓位 | {r.get('max_position', '-') * 100:.0f}% |",
        f"| 止损线 | {r.get('stop_loss', '-') * 100:.0f}% |",
        f"| 止盈线 | {r.get('take_profit', '-') * 100:.0f}% |",
        f"| RSI 超买 | {sig.get('rsi_overbought', '-')} |",
        f"| RSI 超卖 | {sig.get('rsi_oversold', '-')} |",
        f"| MACD 权重 | {sig.get('macd_weight', '-')} |",
        f"| RSI 权重 | {sig.get('rsi_weight', '-')} |",
        f"| 量价权重 | {sig.get('volume_weight', '-')} |",
    ]
    return "\n".join(lines)
 
 
# ─────────────────────────────────────────────
# 数据库查询（统一通过 storage/db.py）
# ─────────────────────────────────────────────
from storage.db import (
    get_latest_signals,
    get_performance_summary,
    get_equity_curve,
)

# ─────────────────────────────────────────────
# 系统 Prompt 构建
# ─────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """你是用户的量化交易助手，负责管理和解释策略配置，并回答市场问题。
 
## 当前策略配置（YAML）
```yaml
{config_yaml}
```
 
## 最新信号（最近5条）
{signals_summary}
 
## 近30天绩效
{performance_summary}

## 北向资金 & 市场情绪
{northbound_sentiment_summary}

## 你的能力
1. **查询**：回答关于当前配置、信号、绩效的任何问题
2. **修改配置**：当用户要求修改参数时，返回如下 JSON 指令（放在回复末尾）：
   <CONFIG_UPDATE>
   {{"key": "risk.stop_loss", "value": 0.04, "reason": "用户指定"}}
   </CONFIG_UPDATE>
3. **分析市场**：结合当前信号和配置给出看法
4. **解释**：解释任何量化指标或策略逻辑
 
## 参数修改规则
- style 只能是：aggressive / balanced / conservative
- max_position 范围：0.1 ~ 1.0
- stop_loss 范围：0.01 ~ 0.10
- take_profit 范围：0.03 ~ 0.30
- rsi_overbought 范围：60 ~ 85
- rsi_oversold 范围：15 ~ 45
- 三个权重之和必须等于 1.0
 
## 回复风格
- 简洁直接，不废话
- 涉及数字时保留2位小数
- 修改配置时必须解释为什么这样改
- 如果用户指令模糊，先询问确认再修改
"""
 
 
def build_system_prompt() -> str:
    config = load_config()
    config_yaml = yaml.dump(config, allow_unicode=True, sort_keys=False)
 
    signals = get_latest_signals(5)
    if signals:
        sig_lines = []
        for s in signals:
            ret_str = f"{s['actual_return']*100:.1f}%" if s.get("actual_return") is not None else "待结算"
            correct_str = "✓" if s.get("is_correct") else ("✗" if s.get("is_correct") is not None else "-")
            sig_lines.append(
                f"  {s['signal_date']} {s['symbol']} {s['direction'].upper()} "
                f"置信度{s['confidence']} 仓位{s['suggested_position']*100:.0f}% "
                f"收益{ret_str} {correct_str}"
            )
        signals_summary = "\n".join(sig_lines)
    else:
        signals_summary = "  暂无历史信号（数据库为空）"
 
    perf = get_performance_summary(30)
    if perf and perf.get("total"):
        performance_summary = (
            f"  总信号数：{perf['total']}  "
            f"胜率：{(perf.get('win_rate') or 0)*100:.1f}%  "
            f"平均收益：{(perf.get('avg_return') or 0)*100:.2f}%  "
            f"最差：{(perf.get('worst') or 0)*100:.2f}%  "
            f"最佳：{(perf.get('best') or 0)*100:.2f}%"
        )
    else:
        performance_summary = "  暂无绩效数据（信号未结算）"
 
    # 北向资金 & 情绪摘要
    nb_sent_lines = []
    try:
        from data.fetcher import get_latest_northbound, get_latest_sentiment
        nb = get_latest_northbound()
        if nb and nb.get('net_flow') is not None:
            direction = '净流入' if nb['net_flow'] > 0 else '净流出'
            nb_sent_lines.append(
                f'  北向资金({nb.get("date", "-")}): {direction} {abs(nb["net_flow"]):.2f} 亿元'
            )
        sent = get_latest_sentiment()
        if sent and sent.get('value'):
            label_map = {
                'extreme_fear': '极度恐惧', 'fear': '恐惧', 'neutral': '中性',
                'greed': '贪婪', 'extreme_greed': '极度贪婪',
            }
            label_cn = label_map.get(sent.get('label', ''), sent.get('label', ''))
            nb_sent_lines.append(
                f'  市场情绪: {sent["value"]:.1f}/100 ({label_cn})'
            )
    except Exception:
        pass

    northbound_sentiment_summary = "\n".join(nb_sent_lines) if nb_sent_lines else "  暂无北向资金和情绪数据"

    return SYSTEM_PROMPT_TEMPLATE.format(
        config_yaml=config_yaml,
        signals_summary=signals_summary,
        performance_summary=performance_summary,
        northbound_sentiment_summary=northbound_sentiment_summary,
    )
 
 
# ─────────────────────────────────────────────
# 解析并执行配置修改指令
# ─────────────────────────────────────────────
def parse_and_apply_config_update(response_text: str) -> Optional[str]:
    """
    从 AI 回复中提取 <CONFIG_UPDATE>...</CONFIG_UPDATE> 块并执行。
    返回修改摘要字符串，无修改返回 None。
    """
    import re
    pattern = r"<CONFIG_UPDATE>\s*(.*?)\s*</CONFIG_UPDATE>"
    match = re.search(pattern, response_text, re.DOTALL)
    if not match:
        return None
 
    try:
        instruction = json.loads(match.group(1))
        key = instruction["key"]
        value = instruction["value"]
 
        config = load_config()
        parts = key.split(".", 1)
        if len(parts) == 2:
            section, param = parts
            if section not in config:
                config[section] = {}
            config[section][param] = value
        else:
            config[key] = value
 
        save_config(config)
        logger.info(f"配置已修改：{key} = {value}")
        return f"✅ 已更新 `{key}` → `{value}`"
 
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"解析配置指令失败：{e}，原文：{match.group(1)}")
        return None
 
 
def clean_response(text: str) -> str:
    """移除回复中的 CONFIG_UPDATE 标签，展示给用户时不显示原始 JSON"""
    import re
    return re.sub(r"\s*<CONFIG_UPDATE>.*?</CONFIG_UPDATE>", "", text, flags=re.DOTALL).strip()
 
 
# ─────────────────────────────────────────────
# 对话核心函数
# ─────────────────────────────────────────────
def chat(message: str, history: list) -> tuple[str, list]:
    """
    处理用户消息，调用 DeepSeek，解析配置修改指令，返回回复。
    兼容 Gradio 4.x：history 格式为 list[dict]，每条为
      {"role": "user"/"assistant", "content": "..."}
    """
    if not message.strip():
        return "", history
 
    try:
        client = get_client()
    except EnvironmentError as e:
        reply = f"❌ {e}\n\n请先执行：\n```\nset DEEPSEEK_API_KEY=你的key\n```"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history
 
    # 构建发给 DeepSeek 的消息列表（system + 历史 + 新消息）
    messages = [{"role": "system", "content": build_system_prompt()}]
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # 兼容旧版 tuple 格式
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    messages.append({"role": "user", "content": message})
 
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.5,
            max_tokens=1000,
        )
        raw_reply = response.choices[0].message.content.strip()
 
        # 解析并执行配置修改
        update_msg = parse_and_apply_config_update(raw_reply)
        display_reply = clean_response(raw_reply)
        if update_msg:
            display_reply = display_reply + f"\n\n{update_msg}"
 
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": display_reply})
        logger.info(f"对话完成，用户：{message[:50]}...")
        return "", history
 
    except Exception as e:
        logger.error(f"DeepSeek 调用失败：{e}")
        reply = f"❌ AI 调用失败：{e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history
 
 
# ─────────────────────────────────────────────
# 面板数据刷新函数
# ─────────────────────────────────────────────
def refresh_config_panel() -> str:
    return config_to_markdown(load_config())
 
 
def refresh_signals_table() -> pd.DataFrame:
    signals = get_latest_signals(20)
    if not signals:
        return pd.DataFrame(
            columns=["日期", "标的", "方向", "置信度", "仓位%", "止损%", "收益%", "正确"]
        )
    rows = []
    for s in signals:
        rows.append({
            "日期": s.get("signal_date", "-"),
            "标的": s.get("symbol", "-"),
            "方向": s.get("direction", "-").upper(),
            "置信度": s.get("confidence", "-"),
            "仓位%": f"{(s.get('suggested_position') or 0)*100:.0f}",
            "止损%": f"{(s.get('stop_loss') or 0)*100:.1f}",
            "收益%": f"{(s.get('actual_return') or 0)*100:.2f}" if s.get("actual_return") is not None else "-",
            "正确": "✓" if s.get("is_correct") == 1 else ("✗" if s.get("is_correct") == 0 else "-"),
        })
    return pd.DataFrame(rows)
 
 
def refresh_equity_chart() -> go.Figure:
    df = get_equity_curve(60)
 
    fig = go.Figure()
 
    if not df.empty:
        # 权益曲线
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            name="累计净值",
            line=dict(color="#1D9E75", width=2),
            fill="tozeroy",
            fillcolor="rgba(29,158,117,0.1)",
        ))
 
        # 回撤着色
        equity = df["equity"]
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
 
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=drawdown,
            mode="lines",
            name="回撤",
            line=dict(color="#D85A30", width=1.5),
            yaxis="y2",
        ))
 
        # 基准线
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      line_width=0.8, annotation_text="基准 1.0")
 
    else:
        # 空数据占位提示
        fig.add_annotation(
            text="暂无绩效数据<br>信号结算后将自动展示",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
 
    fig.update_layout(
        title=dict(text="近60天策略净值与回撤", font=dict(size=14)),
        xaxis=dict(title="日期", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title="累计净值", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis2=dict(title="回撤", overlaying="y", side="right",
                    tickformat=".1%", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig
 
 
def refresh_performance_md() -> str:
    perf = get_performance_summary(30)
    if not perf or not perf.get("total"):
        return "暂无绩效数据（等待信号结算）"
 
    win_rate = (perf.get("win_rate") or 0) * 100
    avg_ret = (perf.get("avg_return") or 0) * 100
    worst = (perf.get("worst") or 0) * 100
    best = (perf.get("best") or 0) * 100
 
    # 简单评分
    score_emoji = "🟢" if win_rate >= 55 and avg_ret > 0 else ("🟡" if win_rate >= 45 else "🔴")
 
    return f"""### 近30天绩效 {score_emoji}
 
| 指标 | 值 |
|---|---|
| 总信号数 | {perf.get('total', 0)} 条 |
| 胜率 | **{win_rate:.1f}%** |
| 平均收益 | **{avg_ret:+.2f}%** |
| 最佳单次 | {best:+.2f}% |
| 最差单次 | {worst:+.2f}% |
"""
 
 
# ─────────────────────────────────────────────
# 手动操作按钮
# ─────────────────────────────────────────────
def manual_fetch_data() -> str:
    try:
        from data.fetcher import update_all, fetch_northbound_flow, fetch_fear_greed_index
        results = update_all()
        nb_count = fetch_northbound_flow(days=60)
        sent = fetch_fear_greed_index()
        ok = sum(1 for v in results.values() if v >= 0)
        added = sum(v for v in results.values() if v > 0)
        parts = [f"行情: {ok}标的新增{added}条"]
        if nb_count >= 0:
            parts.append(f"北向: +{nb_count}条")
        if sent:
            parts.append(f"情绪: {sent['value']:.0f}")
        return f"✅ 数据拉取完成  {' | '.join(parts)}  {datetime.now().strftime('%H:%M:%S')}"
    except ImportError:
        return "⚠️ data/fetcher.py 未找到，请先实现该模块"
    except Exception as e:
        return f"❌ 拉取失败：{e}"
 
 
def manual_generate_signals() -> str:
    try:
        from strategy.signals import run_all
        signals = run_all()
        return f"✅ 已生成 {len(signals)} 条信号  {datetime.now().strftime('%H:%M:%S')}"
    except ImportError:
        return "⚠️ strategy/signals.py 未找到，请先实现该模块"
    except Exception as e:
        return f"❌ 信号生成失败：{e}"
 
 
def manual_optimize() -> str:
    try:
        from strategy.optimizer import run_nightly_optimization
        report = run_nightly_optimization(days=10, dry_run=False)
        action = report.get("action", "unknown")
        reason = report.get("reason", "")
        changes = report.get("changes", {})
        if action == "update":
            change_str = "、".join([f"{k}={v}" for k, v in changes.items()])
            return f"✅ 调优完成：{reason}\n变更：{change_str}"
        else:
            return f"ℹ️ 无需调整：{reason}"
    except ImportError:
        return "⚠️ strategy/optimizer.py 未找到，请先实现该模块"
    except Exception as e:
        return f"❌ 调优失败：{e}"
 
 
def manual_rollback() -> str:
    try:
        from strategy.optimizer import rollback_to_last_backup
        success = rollback_to_last_backup()
        if success:
            return f"✅ 已回滚到上一版本配置  {datetime.now().strftime('%H:%M:%S')}"
        return "⚠️ 无可用备份"
    except ImportError:
        return "⚠️ strategy/optimizer.py 未找到"
    except Exception as e:
        return f"❌ 回滚失败：{e}"
 
 
# ─────────────────────────────────────────────
# 快捷指令列表
# ─────────────────────────────────────────────
QUICK_COMMANDS = [
    "当前策略配置是什么？",
    "切换成保守模式",
    "切换成激进模式",
    "止损改成3%",
    "止损改成6%",
    "最大仓位改成50%",
    "最近30天胜率多少？",
    "今天应该关注哪些信号？",
    "当前RSI超买超卖阈值是多少？",
    "macd权重改成0.5，rsi改成0.3，量价改成0.2",
]
 
 
# ─────────────────────────────────────────────
# 前端 API 数据函数（供 frontend.html 调用）
# fn_index 顺序取决于在 gr.Blocks 中注册的顺序
# ─────────────────────────────────────────────

def api_get_signals_json() -> str:
    """返回最近 50 条信号的 JSON 字符串（数据库为空时自动生成信号）"""
    signals = get_latest_signals(50)
    if not signals:
        try:
            from strategy.signals import run_all
            run_all(dry_run=False)
            signals = get_latest_signals(50)
        except Exception:
            pass
    return json.dumps(signals, ensure_ascii=False, default=str)


def api_get_performance_json() -> str:
    """返回近30天绩效摘要 JSON"""
    perf = get_performance_summary(30)
    return json.dumps(perf, ensure_ascii=False, default=str)


def api_get_config_json() -> str:
    """返回当前 config.yaml 内容 JSON"""
    cfg = load_config()
    return json.dumps(cfg, ensure_ascii=False)


def api_get_equity_json() -> str:
    """返回近60天权益曲线数据 JSON（数据库为空时触发信号生成）"""
    df = get_equity_curve(60)
    if df.empty:
        try:
            from data.fetcher import update_all
            from strategy.signals import run_all
            update_all()
            run_all(dry_run=False)
            df = get_equity_curve(60)
        except Exception:
            pass
    if df.empty:
        return json.dumps([])
    df["date"] = df["date"].astype(str)
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False)


def api_get_sentiment_json() -> str:
    """返回最新市场情绪数据 JSON（数据库为空时自动计算）"""
    try:
        from data.fetcher import get_latest_sentiment, fetch_fear_greed_index
        sent = get_latest_sentiment()
        if not sent or not sent.get("value"):
            sent = fetch_fear_greed_index()
        return json.dumps(sent or {}, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({})


def api_get_status_json() -> str:
    """返回系统运行状态 JSON"""
    try:
        from storage.db import get_db_status
        status = get_db_status()
        return json.dumps(status, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"error": "无法获取状态"})


# ─────────────────────────────────────────────
# 构建 Gradio UI
# ─────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="量化助手",
        theme=gr.themes.Soft(primary_hue="emerald"),
        css="""
        .status-bar { font-size: 12px; color: #888; margin-top: 4px; }
        .quick-btn { font-size: 12px !important; padding: 4px 8px !important; }
        footer { display: none !important; }
        """,
    ) as app:
 
        gr.Markdown("# 📈 量化助手\n*A股策略 · DeepSeek 驱动 · 24h 自动调优*")
 
        with gr.Tabs():
 
            # ══ Tab 1: 对话 ══
            with gr.TabItem("💬 策略对话"):
                with gr.Row():
                    with gr.Column(scale=7):
                        chatbot = gr.Chatbot(
                            label="与助手对话",
                            height=480,
                            show_label=False,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="输入指令，例如：切换成保守模式 / 止损改成4% / 最近胜率多少",
                                show_label=False,
                                scale=8,
                                container=False,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
 
                    with gr.Column(scale=3):
                        gr.Markdown("**快捷指令**")
                        for cmd in QUICK_COMMANDS:
                            gr.Button(cmd, size="sm", elem_classes="quick-btn").click(
                                fn=lambda c=cmd: c,
                                outputs=msg_input,
                            )
 
                # 发送逻辑
                send_btn.click(
                    fn=chat,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                )
                msg_input.submit(
                    fn=chat,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                )
 
            # ══ Tab 2: 策略面板 ══
            with gr.TabItem("📊 策略面板"):
                with gr.Row():
                    refresh_all_btn = gr.Button("🔄 刷新全部", variant="secondary")
 
                with gr.Row():
                    with gr.Column(scale=1):
                        config_md = gr.Markdown(value=refresh_config_panel())
                        perf_md = gr.Markdown(value=refresh_performance_md())
 
                    with gr.Column(scale=2):
                        equity_chart = gr.Plot(value=refresh_equity_chart(), label="净值曲线")
 
                gr.Markdown("### 最近信号记录")
                signals_table = gr.DataFrame(
                    value=refresh_signals_table(),
                    interactive=False,
                )
 
                refresh_all_btn.click(
                    fn=lambda: (
                        refresh_config_panel(),
                        refresh_performance_md(),
                        refresh_equity_chart(),
                        refresh_signals_table(),
                    ),
                    outputs=[config_md, perf_md, equity_chart, signals_table],
                )
 
            # ══ Tab 3: 手动操作 ══
            with gr.TabItem("⚙️ 手动操作"):
                gr.Markdown("""
### 手动触发各阶段任务
 
通常由 `scheduler.py` 自动在特定时间运行，这里可以手动触发一次。
                """)
 
                op_status = gr.Textbox(
                    label="执行结果",
                    interactive=False,
                    lines=3,
                )
 
                with gr.Row():
                    fetch_btn = gr.Button("📥 拉取最新行情", variant="secondary")
                    signal_btn = gr.Button("📡 生成今日信号", variant="secondary")
                    optimize_btn = gr.Button("🤖 AI 参数调优", variant="primary")
                    rollback_btn = gr.Button("↩️ 回滚配置", variant="stop")
 
                fetch_btn.click(fn=manual_fetch_data, outputs=op_status)
                signal_btn.click(fn=manual_generate_signals, outputs=op_status)
                optimize_btn.click(fn=manual_optimize, outputs=op_status)
                rollback_btn.click(fn=manual_rollback, outputs=op_status)
 
                gr.Markdown("""
---
### 调优说明
 
- **拉取行情**：调用 `data/fetcher.py`，更新本地数据库
- **生成信号**：运行因子计算，写入今日信号
- **AI 参数调优**：分析近10天绩效，让 DeepSeek 建议参数变更（自动备份旧配置）
- **回滚配置**：恢复到上一次调优前的参数版本
                """)
 
        # 页脚状态
        with gr.Row():
            gr.Markdown(
                f"<div class='status-bar'>启动时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"· 配置文件：{CONFIG_PATH} "
                f"· 数据库：{DB_PATH}</div>"
            )
    return app


# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='量化助手 Web 界面')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址（默认 127.0.0.1）')
    parser.add_argument('--port', type=int, default=7860, help='端口（默认 7860）')
    parser.add_argument('--share', action='store_true', help='生成公网临时链接（Gradio share）')
    args = parser.parse_args()

    # ── 1. 创建自定义 FastAPI 应用并注册 API 路由 ──
    from gradio.routes import App
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import fastapi

    custom_app = App()
    custom_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @custom_app.get("/frontend")
    async def serve_frontend():
        frontend_path = Path(__file__).parent / "frontend.html"
        if frontend_path.exists():
            return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h3>frontend.html 未找到</h3>", status_code=404)

    @custom_app.get("/api/signals")
    async def api_signals():
        return JSONResponse(content=json.loads(api_get_signals_json()))

    @custom_app.get("/api/performance")
    async def api_performance():
        return JSONResponse(content=json.loads(api_get_performance_json()))

    @custom_app.get("/api/config")
    async def api_config():
        return JSONResponse(content=json.loads(api_get_config_json()))

    @custom_app.get("/api/equity")
    async def api_equity():
        return JSONResponse(content=json.loads(api_get_equity_json()))

    @custom_app.get("/api/sentiment")
    async def api_sentiment():
        return JSONResponse(content=json.loads(api_get_sentiment_json()))

    @custom_app.get("/api/status")
    async def api_status():
        return JSONResponse(content=json.loads(api_get_status_json()))

    @custom_app.post("/api/chat")
    async def api_chat(request: fastapi.Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"reply": "无法解析请求，请刷新页面后重试"})

        msg = body.get("message", "").strip()
        history = body.get("history", [])

        logger.info(f"Chat API: msg={msg[:30]}... history_len={len(history)}")

        if not msg:
            return JSONResponse(content={"reply": "请输入消息内容"})

        try:
            _, new_history = chat(msg, history)
        except Exception as e:
            logger.error(f"Chat 调用失败: {e}")
            return JSONResponse(content={"reply": f"对话服务异常: {str(e)[:200]}"})

        if not new_history:
            return JSONResponse(content={"reply": "对话处理失败，请稍后重试"})

        last = new_history[-1]
        reply = last.get("content", "") if isinstance(last, dict) else str(last)

        if not reply:
            reply = "（系统未生成有效回复，请检查 API Key 配置）"

        return JSONResponse(content={"reply": reply})

    logger.info("API 路由已注册: /api/signals, /api/performance, /api/config, /api/equity, /api/sentiment, /api/chat, /frontend")

    # ── 2. 构建 Gradio UI ──
    app_instance = build_ui()

    # ── 3. 启动 ──
    frontend_url = f"http://{args.host}:{args.port}/frontend"
    print("=" * 50)
    print(f"  仪表盘地址: {frontend_url}")
    print(f"  Gradio 界面: http://{args.host}:{args.port}")
    print("=" * 50)

    app_instance.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        strict_cors=False,
        _app=custom_app,
    )
