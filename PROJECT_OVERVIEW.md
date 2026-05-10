# 量化助手 — 项目概览

A股量化交易助手，通过 Hermes Agent 连接进行管理。APScheduler 驱动的 24h 自动运行循环，技术因子计算 → 规则评分直出决策，新浪实时行情兜底，盘中监控+止损微信推送。

> ⚠️ 本项目通过 Hermes 进行配置、调优和异常处理。没有 Hermes 时系统按最后配置自动运行，但停止工作后需通过 Hermes 排查修复。

---

## 架构总览

```
                        ┌─────────────────────────────┐
                        │   scheduler.py (APScheduler) │  ← 24h 主调度器
                        │   ┌─ 09:00 自动拉取行情     │
                        │   ├─ 09:15 自动生成信号      │
                        │   ├─ 盘中 自动轮询监控      │
                        │   └─ 周五22:00 自动参数调优  │
                        └─────────┬───────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼──┐   ┌─────▼──┐   ┌─────▼────┐
              │ fetch  │   │ signals│   │ optimizer │
              │ er.py  │   │ .py    │   │ .py       │
              └───┬────┘   └───┬────┘   └─────┬─────┘
                  │            │              │
              ┌───▼────────────▼──────────────▼────┐
              │        SQLite (db/quant.db)        │
              │ market_data / signals /             │
              │ sentiment_index / config_versions   │
              └────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ monitor_daemon.py │  ← 盘中实时监控
                    │ 新浪轮询 + 微信推  │
                    └───────────────────┘
```

### 启动方式

```bash
# 【推荐】完整模式
python scheduler.py
# → 自动完成：行情拉取 → 信号生成 → 盘中监控 → 周五调优

# 仅盘中监控
python monitor/monitor_daemon.py

# 仅生成信号
python strategy/signals.py
```

---

## 工作流程

### 每日自动循环

```
09:00 ── data/fetcher.update_all()
           ├── 新浪 hq.sinajs.cn 拉取行情 (OHLCV)  ← 首选
           ├── 腾讯 web.sqt.gtimg.cn 备选
           ├── INSERT OR IGNORE → market_data 表
           └── 增量更新（跳过已有日期）

09:15 ── strategy/signals.run_all()
           ├── 读 market_data → 计算技术因子
           ├── 规则评分直出信号（纯代码，无 LLM）
           ├── MACD(40%) + RSI(30%) + 量价(30%) + 布林补正(±5)
           ├── 四因子加权合成 0-100 分 → buy/sell/hold
           └── INSERT OR REPLACE → signals 表

盘中 ── monitor/monitor_daemon.py
           ├── 交易日 9:15-15:05 自动启动
           ├── 每 30 秒新浪拉取持仓最新价
           ├── 跌破止损线 → 自动推送微信提醒
           └── 每只标的每交易日仅触发一次，防刷屏

周五 22:00 ── strategy/optimizer.run_weekly_optimization()
           ├── 读近期 signals 绩效（含 IC/胜率）
           ├── 80/20 时序分割遍历候选阈值
           ├── 验证集 IC 未改善 → 自动跳过（防误调）
           ├── 备份旧配置 → diff 写入新 config.yaml
           └── 记录 config_versions 表
```

---

## 数据源

| 数据 | 来源 | 存储表 | 更新频率 | 说明 |
|------|------|--------|----------|------|
| A股日线行情 | 新浪 `hq.sinajs.cn`（首选）+ 腾讯 `web.sqt.gtimg.cn`（备选） | `market_data` | 每交易日 | 东财 push2 API 已被封（`HTTP 000`），使用新浪。新浪返回 GBK 编码，需 `resp.encoding='gbk'` 后解析 |
| 实时行情（盘中监控） | 新浪 `hq.sinajs.cn` | — | 30秒轮询 | 涨跌幅/量比/振幅，新浪无 volume_ratio/speed_5m 等字段，缺失字段降级处理 |
| 北向资金 | AKShare（数据截至 2024-08） | `northbound_flow` | 手动 | 沪港通/深港通净流入历史 |
| 融资融券 | AKShare 沪深交易所 | `margin_data` | 每交易日 | 融资余额变化，杠杆资金态度 |
| 市场情绪 | 自行计算（2维简化版） | `sentiment_index` | 每交易日 | 涨跌幅+量能加权（2026-05-07简化） |
| 交易信号 | 规则评分直出（无 LLM） | `signals` | 每交易日 | 含方向/置信度/仓位/止损/基准涨跌幅 |
| 策略配置 | `config.yaml` | `config_versions` | 调优时 | 版本化管理，支持回滚 |

### 数据合理性验证

每次拉取行情后自动执行（`data/realtime.py`）：
- 涨跌幅范围：`-50% < pct < +50%` → 超出标记为垃圾数据，需 Hermes 介入确认
- 价格正数：`price > 0`
- 交叉验证：新浪异常时自动用腾讯接口二次确认
- 假阳性过滤：盘后数据更新时跳过非交易日代理指标

---

## 技术因子（strategy/signals.py）

所有因子计算纯 pandas 实现，不依赖 TA-Lib 或任何外部 API。

### MACD（权重 40%）

```
EMA_fast = close.ewm(span=12).mean()
EMA_slow = close.ewm(span=26).mean()
DIF = EMA_fast - EMA_slow,  DEA = DIF.ewm(span=9).mean()
hist = (DIF - DEA) × 2

→ hist > 0 → 看涨 (60-100 分)
→ hist < 0 → 看跌 (0-40 分)
→ 金叉 +15 分 / 死叉 -15 分
```

### RSI（权重 30%）

```
delta = close.diff()
gain = delta.clip(lower=0),  loss = (-delta).clip(lower=0)
avg_g = gain.ewm(com=13).mean(),  avg_l = loss.ewm(com=13).mean()
RSI = 100 - 100 / (1 + avg_g/avg_l)

→ RSI ≤ 超卖线(30) → 看涨 (75-100 分)
→ RSI ≥ 超买线(70) → 看跌 (0-25 分)
→ 中性区线性映射 (30-70 分)
```

### 布林带（补正 ±5 分）

```
MA = close.rolling(20).mean()
std = close.rolling(20).std()
上轨 = MA + 2×std,  下轨 = MA - 2×std
position = (close - 下轨) / (上轨 - 下轨)

→ position < 0.1 触下轨 → 超卖补正 +5
→ position > 0.9 触上轨 → 超买补正 -5
```

### 量价因子（权重 30%）

```
量比 = volume[-1] / mean(volume[-21:-1])
OBV = cumsum(volume × sign(close.diff()))

→ 量比 > 1.5 + OBV 向上 → 看涨 (75 分)
→ 量比 > 1.2 + OBV 向上 → 看涨 (65 分)
→ 量比 < 0.7 缩量 → 看跌 (35 分)
→ OBV 向下 → 看跌 (40 分)
```

### 综合评分 & 决策

```
score = MACD×40% + RSI×30% + 量价×30% + 布林补正
score ∈ [0, 100]

buy_threshold = config.signals.buy_threshold（默认63）
sell_threshold = config.signals.sell_threshold（默认37）

≥ buy_threshold → buy
≤ sell_threshold → sell
buy_threshold > score > sell_threshold → hold, 仓位 0

（阈值由 optimizer.py 每周五 80/20 时序分割自动调优，IC 不改善自动跳过）
```

---

## 盘中实时监控 & 止损推送

由 `monitor/monitor_daemon.py` 守护进程实现：

- **交易日自动启停**：系统 crontab `15 9 * * 1-5` 启动，`5 15 * * 1-5` 清理
- 每 **30 秒**通过新浪轮询所有持仓标的最新价
- 跌破 5% 止损线 → 自动推送微信消息（每只标的每交易日仅触发一次）
- 跌破 4.5% 提醒线 → 预警消息
- 超过 15:05 非交易时段自动退出

### 当前持仓

| 代码 | 名称 | 成本价 | 配比 | 止损线(5%) | 提醒线(4.5%) |
|------|------|--------|------|-----------|-------------|
| 600900 | 长江电力 | 27.15 | 50% | 25.79 | 25.93 |
| 159903 | 深成ETF南方 | 1.878 | 30% | 1.784 | 1.793 |
| 002274 | 华昌化工 | 6.89 | 20% | 6.55 | 6.58 |

---

## 参数调优（strategy/optimizer.py，每周五自动执行）

1. 查询近期 signals 绩效（胜率/IC/平均收益/方向分布）
2. 80/20 时序分割遍历候选阈值
3. IC 未改善自动跳过，绝不盲改参数
4. 新参数优于旧参数 → 自动备份旧配置 → 写入新配置
5. 支持 `--dry-run` 预览不写入

---

## IC 监控（strategy/ic_monitor.py）

每日自动计算信号 composite_score 与次日实际收益之间的 RankIC：
- 连续 5 日 RankIC 为负 → 自动发出退化告警（日志 + 微信推送）
- Hermes 收到告警后介入分析退化原因

---

## 与 Hermes 的交互

虽然是全自动化系统，但以下场景需要联系 Hermes：

| 场景 | 说明 |
|------|------|
| ❌ 系统停止运行 | scheduler 或 monitor 进程意外退出，需 Hermes 排查重启 |
| ⚠️ IC 退化告警 | RankIC 连续为负，需 Hermes 分析因子有效性 |
| 🔧 策略调整 | 修改持仓、调参数、换标的，通过 Hermes 操作 config.yaml |
| 📊 数据异常 | 行情接口变更/被封、垃圾数据，需 Hermes 修复数据源 |
| 📋 复盘分析 | 回顾信号历史表现、回测结果，通过 Hermes 读取数据库 |

---

## 快速开始

```bash
# 安装依赖（一次性）
pip install akshare pyyaml plotly pandas "httpx<0.28" apscheduler

# 启动
python scheduler.py

# 手动执行单项任务
python data/fetcher.py --all           # 拉取行情
python strategy/signals.py             # 生成信号
python monitor/monitor_daemon.py       # 启动盘中监控
python strategy/optimizer.py --dry-run  # 预览参数调优
```

---

## 项目文件结构

```
/opt/quant-trader/
├── config.yaml              ← 策略配置（仓位/止损/阈值等，版本化管理）
├── PROJECT_OVERVIEW.md      ← 本文件
├── CONTEXT.md               ← Hermes 自动读取的项目上下文
├── scheduler.py             ← 主调度器（APScheduler）
├── data/
│   ├── fetcher.py           ← 行情拉取（新浪+腾讯）
│   └── realtime.py          ← 实时数据代理 + 合理性验证
├── strategy/
│   ├── signals.py           ← 技术因子计算 + 信号生成（核心逻辑）
│   ├── optimizer.py         ← 每周五参数调优
│   ├── ic_monitor.py        ← IC 退化监控
│   └── intraday_monitor.py  ← 盘内信号辅助
├── monitor/
│   └── monitor_daemon.py    ← 盘中监控守护进程
├── db/
│   └── quant.db             ← SQLite 数据库（自动创建）
├── logs/                    ← 运行日志
└── config_backups/          ← 配置版本备份
```

---

*最后更新: 2026-05-10*
