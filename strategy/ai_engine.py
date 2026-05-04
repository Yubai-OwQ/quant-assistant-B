"""
strategy/ai_engine.py - DeepSeek AI 决策引擎
负责市场分析、策略生成、夜间参数调优
"""
import json
import logging
import os
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(messages: list, config: dict, temperature: float = None) -> str:
    """调用 DeepSeek API，返回文本响应"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "【未配置 DEEPSEEK_API_KEY，请在环境变量中设置】"

    ds_cfg = config.get("deepseek", {})
    payload = {
        "model": ds_cfg.get("model", "deepseek-chat"),
        "messages": messages,
        "max_tokens": ds_cfg.get("max_tokens", 2000),
        "temperature": temperature if temperature is not None else ds_cfg.get("temperature", 0.3),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.error("DeepSeek API 超时")
        return "API 调用超时，请稍后重试"
    except Exception as e:
        logger.error(f"DeepSeek API 错误: {e}")
        return f"API 调用失败: {str(e)}"


def analyze_market(signals: dict, fear_greed: dict, config: dict,
                   northbound: dict = None) -> dict:
    """
    主分析函数：输入技术信号 + 情绪数据 + 北向资金，输出策略建议。
    返回结构化字典。
    """
    mode = config.get("strategy", {}).get("mode", "balanced")
    market = config.get("strategy", {}).get("market", "A股")
    risk_cfg = config.get("risk", {})

    mode_desc = {
        "aggressive": "激进模式：追求高收益，可接受较大回撤，仓位偏高，止损宽松",
        "balanced": "均衡模式：收益与风险平衡，中等仓位，适中止损",
        "conservative": "保守模式：资本保全优先，低仓位，严格止损，宁可错过也不追涨"
    }.get(mode, "均衡模式")

    system_prompt = f"""你是一个专业的量化策略分析师，专注于{market}市场。
当前策略风格：{mode_desc}

你的任务是基于技术信号、市场情绪和北向资金流向，给出具体可执行的交易建议。
北向资金（外资通过沪深港通流入A股）是重要的市场风向标：
- 持续净流入 → 外资看多，可增强做多信心
- 持续净流出 → 外资撤离，应提高警惕
- 单日大额流入/流出需结合趋势判断

请严格以JSON格式输出，不要有任何额外文字。

JSON格式要求：
{{
  "direction": "做多/做空/观望",
  "confidence": 0.0到1.0之间的数字,
  "suggested_position": 0.0到1.0之间（占总资金比例）,
  "stop_loss_pct": 建议止损百分比（正数）,
  "take_profit_pct": 建议止盈百分比（正数）,
  "reasoning": "简短的中文分析（100字以内）",
  "key_risks": "主要风险点（50字以内）",
  "hold_days": 建议持仓天数（整数）
}}"""

    # 构建北向资金信息
    nb_info = ""
    if northbound and northbound.get("net_flow") is not None:
        nb_flow = northbound.get("net_flow", 0)
        nb_dir = "净流入" if nb_flow > 0 else "净流出"
        nb_daily = f"当日{nb_dir} {abs(nb_flow):.2f} 亿元"
        nb_cum = northbound.get("cumulative_flow", 0)
        nb_info = f"""
北向资金：
- {nb_daily}
- 历史累计净流入: {nb_cum:.2f} 亿元"""

    user_content = f"""当前市场数据（{datetime.now().strftime('%Y-%m-%d %H:%M')}）：

技术信号：
- 综合信号: {signals.get('composite', 0):.3f}（-1极度看空，+1极度看多）
- MACD信号: {signals.get('macd', 0):.3f}
- RSI值: {signals.get('rsi_value', 50):.1f}
- 布林带信号: {signals.get('bollinger', 0):.3f}
- 成交量异动: {signals.get('volume', 0):.3f}
- 短期趋势: {signals.get('trend', 0):.3f}
- 当前价格变动: {signals.get('price_change_pct', 0):.2f}%
{nb_info}
市场情绪：
- 综合情绪指数: {fear_greed.get('value', 50):.1f} / 100（{fear_greed.get('label', '中性')}）
- 细分：北向={fear_greed.get('northbound_score', 50):.0f} 资金={fear_greed.get('fund_flow_score', 50):.0f} 宽度={fear_greed.get('breadth_score', 50):.0f} 动量={fear_greed.get('momentum_score', 50):.0f}

风险参数上限：
- 最大仓位: {risk_cfg.get('max_position', 0.8) * 100:.0f}%
- 最大止损: {risk_cfg.get('stop_loss', 0.05) * 100:.0f}%

请基于以上数据给出{mode_desc.split('：')[0]}的交易建议："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    raw = call_deepseek(messages, config)
    return _parse_json_response(raw, signals, config)


def _parse_json_response(raw: str, signals: dict, config: dict) -> dict:
    """解析AI返回的JSON，容错处理"""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            result["raw_response"] = raw
            result["timestamp"] = datetime.now().isoformat()
            return result
    except Exception as e:
        logger.warning(f"JSON解析失败: {e}，使用默认值")

    mode = config.get("strategy", {}).get("mode", "balanced")
    comp = signals.get("composite", 0)
    direction = "做多" if comp > 0.2 else ("做空" if comp < -0.2 else "观望")
    pos = {"aggressive": 0.7, "balanced": 0.5, "conservative": 0.3}.get(mode, 0.5)
    return {
        "direction": direction,
        "confidence": abs(comp),
        "suggested_position": pos if direction != "观望" else 0.0,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 12.0,
        "reasoning": f"综合信号{comp:.2f}，{direction}",
        "key_risks": "AI解析失败，使用默认风控参数",
        "hold_days": 3,
        "raw_response": raw,
        "timestamp": datetime.now().isoformat()
    }


def nightly_optimize(performance_history: list, current_config: dict) -> dict:
    """
    夜间AI调优：基于历史绩效，建议新的策略参数
    返回建议修改的参数字典
    """
    if len(performance_history) < 3:
        return {"message": "历史数据不足，暂不调优", "changes": {}}

    recent = performance_history[-7:]
    avg_sharpe = sum(r.get("sharpe", 0) for r in recent) / len(recent)
    avg_dd = sum(r.get("max_drawdown", 0) for r in recent) / len(recent)
    avg_wr = sum(r.get("win_rate", 0.5) for r in recent) / len(recent)

    system_prompt = """你是量化策略调优专家。基于最近的绩效数据，建议调整策略参数。
请严格以JSON格式输出，格式如下：
{
  "assessment": "一句话评估（50字以内）",
  "changes": {
    "signals.rsi_period": 新值（可选）,
    "signals.macd_fast": 新值（可选）,
    "signals.bb_std": 新值（可选）,
    "risk.stop_loss": 新值（可选）,
    "risk.max_position": 新值（可选）
  },
  "reasoning": "调整理由（100字以内）"
}
只建议有必要改动的参数，不需要改的不要包含在changes中。"""

    user_content = f"""最近7天绩效统计：
- 平均Sharpe比率: {avg_sharpe:.3f}（>1为良好）
- 平均最大回撤: {avg_dd * 100:.2f}%
- 平均胜率: {avg_wr * 100:.1f}%

当前参数：
- RSI周期: {current_config.get('signals', {}).get('rsi_period', 14)}
- MACD快线: {current_config.get('signals', {}).get('macd_fast', 12)}
- 布林带标准差倍数: {current_config.get('signals', {}).get('bb_std', 2.0)}
- 止损比例: {current_config.get('risk', {}).get('stop_loss', 0.05) * 100:.1f}%
- 最大仓位: {current_config.get('risk', {}).get('max_position', 0.8) * 100:.0f}%
- 当前策略模式: {current_config.get('strategy', {}).get('mode', 'balanced')}

请给出参数调优建议："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    raw = call_deepseek(messages, current_config, temperature=0.2)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            result["timestamp"] = datetime.now().isoformat()
            return result
    except Exception:
        pass

    return {"message": raw, "changes": {}, "timestamp": datetime.now().isoformat()}


def chat_with_strategy(user_message: str, current_config: dict,
                       latest_signals: dict, latest_analysis: dict,
                       chat_history: list) -> tuple:
    """
    对话式策略调整接口
    返回 (AI回复文本, 更新后的config)
    """
    system_prompt = f"""你是一个量化交易助手，帮助用户管理和调整交易策略。

当前策略状态：
- 市场：{current_config.get('strategy', {}).get('market', 'A股')}
- 模式：{current_config.get('strategy', {}).get('mode', 'balanced')}
- 最大仓位：{current_config.get('risk', {}).get('max_position', 0.8) * 100:.0f}%
- 止损比例：{current_config.get('risk', {}).get('stop_loss', 0.05) * 100:.0f}%
- 综合信号：{latest_signals.get('composite', 0):.3f}
- AI建议方向：{latest_analysis.get('direction', '未知')}
- AI置信度：{latest_analysis.get('confidence', 0) * 100:.0f}%

你可以帮助用户：
1. 切换策略模式（激进/均衡/保守）
2. 调整风险参数（止损、仓位）
3. 解释当前市场信号和AI建议
4. 查询策略绩效

如果用户要求修改参数，请在回复末尾加上 JSON 标记，格式：
[CONFIG_UPDATE]{{"strategy.mode": "...", "risk.stop_loss": 0.05}}[/CONFIG_UPDATE]

用中文回复，简洁专业。"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history[-6:]:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_message})

    raw = call_deepseek(messages, current_config, temperature=0.5)

    updated_config = current_config.copy()
    clean_response = raw

    if "[CONFIG_UPDATE]" in raw and "[/CONFIG_UPDATE]" in raw:
        try:
            start = raw.find("[CONFIG_UPDATE]") + len("[CONFIG_UPDATE]")
            end = raw.find("[/CONFIG_UPDATE]")
            update_json = json.loads(raw[start:end])
            clean_response = raw[:raw.find("[CONFIG_UPDATE]")].strip()

            import copy
            updated_config = copy.deepcopy(current_config)
            for key_path, value in update_json.items():
                parts = key_path.split(".")
                d = updated_config
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = value
                logger.info(f"配置更新: {key_path} = {value}")

            clean_response += f"\n\n✅ 配置已更新：{update_json}"
        except Exception as e:
            logger.warning(f"配置解析失败: {e}")

    return clean_response, updated_config
