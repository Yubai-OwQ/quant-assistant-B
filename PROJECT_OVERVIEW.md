# 量化助手 — 项目概览

A股量化交易助手，APScheduler 驱动的 24h 自动运行循环。技术因子计算 → 规则评分决策（无 LLM），新浪实时行情兜底，盘中监控+止损推送。

---

## 架构总览

```
                        ┌─────────────────────────────┐
                        │   scheduler.py (APScheduler) │  ← 24h 主调度器
                        │   ┌─ 09:00 拉取行情         │
                        │   ├─ 09:15 生成信号          │
                        │   ├─ 盘中 轮询监控           │
                        │   └─ 周五22:00 参数调优      │
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
# 调度器 + Gradio 界面
python scheduler.py

# 仅盘中监控守护进程
bash start_monitor.sh        # 或手动运行：
python monitor/monitor_daemon.py
```

`scheduler.py` 启动 APScheduler 后台调度 + Gradio 对话界面，所有定时任务在调度器内注册。

---

## 工作流程

### 每日自动循环（由 scheduler.py + APScheduler 驱动）

```
09:00 ── data/fetcher.update_all()
           ├── 新浪 hq.sinajs.cn 拉取行情 (OHLCV)  ← 首选
           ├── 腾讯 web.sqt.gtimg.cn 备选
           ├── INSERT OR IGNORE → market_data 表
           └── 增量更新（跳过已有日期）

09:15 ── strategy/signals.run_all()
           ├── 读 market_data → 计算技术因子
           ├── 规则评分直出信号（无 DeepSeek/LLM 决策层）
           ├── MACD(40%) + RSI(30%) + 量价(30%) + 布林补正(±5)
           ├── 四因子加权合成 0-100 分 → buy/sell/hold
           └── INSERT OR REPLACE → signals 表

盘中 ── monitor/monitor_daemon.py
           ├── 交易日 9:15-15:05 守护进程轮询
           ├── 每 30 秒新浪拉取持仓最新价
           ├── 跌破止损线 → 推微信提醒（每只每日仅一次）
           └── crontab: 15 9 启动 / 5 15 清理

周五 22:00 ── strategy/optimizer.run_weekly_optimization()
           ├── 读近期 signals 绩效（含 IC/胜率）
           ├── 80/20 时序分割遍历候选阈值
           ├── 验证集 IC 未改善 → 跳过（防护）
           ├── 备份旧配置 → diff 写入新 config.yaml
           └── 记录 config_versions 表
```

---

## 数据源

| 数据 | 来源 | 存储表 | 更新频率 | 说明 |
|------|------|--------|----------|------|
| A股日线行情 | 新浪 `hq.sinajs.cn`（首选）+ 腾讯 `web.sqt.gtimg.cn`（备选） | `market_data` | 每交易日 | 东财 push2 API 已被封（`HTTP 000`），改用新浪兜底。新浪返回 GBK 编码，需 `resp.encoding='gbk'` 后解析 |
| 实时行情（盘中监控） | 新浪 `hq.sinajs.cn` | — | 30秒轮询 | 涨跌幅/量比/振幅，新浪无 volume_ratio/speed_5m 等字段，缺失字段降级处理 |
| 北向资金 | AKShare（数据截至 2024-08） | `northbound_flow` | 手动 | 沪港通/深港通净流入历史 |
| 融资融券 | AKShare 沪深交易所 | `margin_data` | 每交易日 | 融资余额变化，杠杆资金态度 |
| 市场情绪 | 自行计算（2维简化版） | `sentiment_index` | 每交易日 | 涨跌幅+量能加权（2026-05-07简化） |
| 交易信号 | 规则评分直出（无 LLM） | `signals` | 每交易日 | 含方向/置信度/仓位/止损/基准涨跌幅 |
| 策略配置 | `config.yaml` | `config_versions` | 调优时 | 版本化管理，支持回滚 |

### 数据合理性验证

每次拉取行情后做以下检查（参见 `data/realtime.py`）：
- **涨跌幅范围**：`-50% < pct < +50%`，超出则标记为垃圾数据
- **价格正数**：`price > 0`
- **交叉验证**：新浪数据异常时用腾讯接口二次确认
- **假阳性过滤**：盘后数据更新时跳过非交易日代理指标

---

## 技术因子（strategy/signals.py）

### MACD（权重 40%）

```
EMA_fast = close.ewm(span=12).mean()
EMA_slow = close.ewm(span=26).mean()
DIF = EMA_fast - EMA_slow
DEA = DIF.ewm(span=9).mean()
hist = (DIF - DEA) × 2

→ hist > 0 → 看涨 (60-100 分)
→ hist < 0 → 看跌 (0-40 分)
→ 金叉 +15 分 / 死叉 -15 分
```

### RSI（权重 30%）

```
delta = close.diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_g = gain.ewm(com=13).mean()
avg_l = loss.ewm(com=13).mean()
RSI = 100 - 100 / (1 + avg_g/avg_l)

→ RSI ≤ 超卖线(30) → 看涨 (75-100 分)
→ RSI ≥ 超买线(70) → 看跌 (0-25 分)
→ 中性区线性映射 (30-70 分)
```

### 布林带（补正 ±5 分）

```
MA = close.rolling(20).mean()
std = close.rolling(20).std()
上轨 = MA + 2×std
下轨 = MA - 2×std
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

≥ buy_threshold → buy,  仓位 = min(max_pos, max(0, max_pos × (score-buy_t+3)/40))
≤ sell_threshold → sell, 仓位同上
buy_threshold > score > sell_threshold → hold, 仓位 0

（阈值由 optimizer.py 每周五 80/20 时序分割调优，验证集 IC 不改善则跳过）
```

### KDJ / OBV（辅助参考）

KDJ、OBV 作为辅助指标参与信号生成，不直接计入评分权重。

---

## 市场情绪指数

### 盘中实时情绪代理（signals.py `_calc_market_sentiment`）

2026-05-07 简化版，基于实时行情数据推算，不需要外部新闻源：

| 维度 | 权重 | 数据来源 |
|------|------|----------|
| 涨跌幅 | 50% | 新浪实时行情 |
| 量能比 | 50% | 量比粗估（无量比时用成交额推断） |

综合评分映射：≥75 亢奋 / ≥60 偏多 / ≥40 中性 / ≥25 偏空 / <25 恐慌

### 标签区间

```
extreme_fear  (0-25)  → 🔴 极度恐惧
fear          (25-45) → 🟠 恐惧
neutral       (45-55) → ⚪ 中性
greed         (55-75) → 🟢 贪婪
extreme_greed (75-100) → 💚 极度贪婪
```

---

## 盘中实时监控 & 止损推送

由 `monitor/monitor_daemon.py` 守护进程实现，交易日 9:15-15:05 运行：

- 每 **30 秒**通过新浪轮询所有持仓标的最新价
- 计算当日涨跌幅，与 `config.yaml` 中 `monitor.stop_loss_watch` 的成本价对比
- 跌破 5% 止损线 → 输出微信推送格式消息 → 推送到用户微信
- 跌破 4.5% 提醒线 → 输出预警消息
- **每只标的每交易日仅触发一次**，避免刷屏
- 超过 15:05 非交易时段自动退出轮询
- 系统 crontab 自动启停：`15 9 * * 1-5` 启动，`5 15 * * 1-5` 清理

**当前持仓（2026-05-07确认）：**

| 代码 | 名称 | 成本价 | 配比 | 止损线(5%) |
|------|------|--------|------|-----------|
| 600900 | 长江电力 | 27.15 | 50% | 25.79 |
| 159903 | 深成ETF南方 | 1.878 | 30% | 1.784 |
| 002274 | 华昌化工 | 6.89 | 20% | 6.55 |

盘中监控配置见 `config.yaml` 的 `monitor` 段。

---

## 数据库表结构

### market_data（行情）

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | TEXT | 标的代码 |
| date | TEXT | 交易日 (PRIMARY KEY) |
| open/high/low/close | REAL | 价格 |
| volume/amount | REAL | 成交量/额 |
| pct_change | REAL | 涨跌幅% |

### signals（信号）

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | TEXT | 标的 |
| signal_date | TEXT | 信号日期 |
| direction | TEXT | buy/sell/hold |
| confidence | INTEGER | 置信度 (0-100) |
| suggested_position | REAL | 建议仓位 (0-1) |
| stop_loss | REAL | 止损百分比 |
| composite_score | REAL | 综合评分 |
| actual_return | REAL | 实际收益（次日结算） |
| is_correct | INTEGER | 方向判断是否正确 |
| benchmark_return | REAL | 沪深300当日涨跌幅（性能基准） |

### northbound_flow（北向资金）

| 字段 | 类型 | 说明 |
|------|------|------|
| date | TEXT | 交易日 (PRIMARY KEY) |
| net_flow | REAL | 当日净流入（亿元） |
| buy_amount | REAL | 买入成交额 |
| sell_amount | REAL | 卖出成交额 |
| cumulative_flow | REAL | 历史累计净买额 |

### margin_data（融资融券）

| 字段 | 类型 | 说明 |
|------|------|------|
| date | TEXT | 交易日 (PRIMARY KEY) |
| sh_margin_balance | REAL | 沪市融资余额 |
| sz_margin_balance | REAL | 深市融资余额 |
| total_margin | REAL | 两市融资余额合计 |
| sh_margin_inflow | REAL | 沪市融资买入额 |
| sz_margin_inflow | REAL | 深市融资买入额 |

### sentiment_index（情绪指数）

| 字段 | 类型 | 说明 |
|------|------|------|
| date | TEXT | 交易日 (PRIMARY KEY) |
| value | REAL | 综合情绪 (0-100) |
| label | TEXT | 标签 |
| margin_score | REAL | 融资情绪得分 |
| fund_flow_score | REAL | 主力资金得分 |
| basis_score | REAL | 期货基差得分 |
| breadth_score | REAL | 市场宽度得分 |
| momentum_score | REAL | 动量得分 |

### config_versions（配置版本）

| 字段 | 类型 | 说明 |
|------|------|------|
| version | INTEGER | 版本号 |
| config_json | TEXT | 完整配置快照 |
| change_reason | TEXT | 变更原因 |

---

## 参数调优（strategy/optimizer.py）

每周五 22:00 自动执行（scheduler.py 配置）：

1. 查询近期 signals 绩效（胜率/IC/平均收益/方向分布）
2. 80/20 时序分割：前 80% 数据训练候选阈值，后 20% 验证
3. 遍历候选参数组合：`buy_threshold ∈ {58,60,63,65,68,70}`，`sell_threshold = 100 - buy_threshold`
4. **验证集 IC 未改善** → 跳过本次调优（不修改 config.yaml）
5. 新参数优于旧参数 → 备份旧配置 → 写入新配置 → 记录 config_versions
6. 新旧参数并行运行一周静默对比（logging 层面，不下发新信号）
7. 支持 `--dry-run` 预览不写入

### 调优原则
- 胜率 < 45% → 优先收紧止损，降低仓位上限
- 平均收益 < 0% → 调整 RSI 超买超卖阈值
- 胜率 > 65% 且收益 > 0 → 适当放宽仓位上限
- 单次参数变动 ≤ 当前值 20%（防止跳变）

---

## IC 监控（strategy/ic_monitor.py）

独立监控模块，每日计算信号 composite_score 与次日实际收益之间的 IC（RankIC）：

- **RankIC**：信号评分与次日涨跌幅的斯皮尔曼秩相关系数
- 连续 5 日 RankIC 为负 → 发出退化告警（日志 + 微信推送）
- 阈值可配置（默认连续负值天数 ≥ 5）

---

## 快速开始

```bash
# 安装依赖
pip install akshare gradio pyyaml plotly pandas "httpx<0.28" apscheduler

# 设置环境变量（如用到 DeepSeek，当前已不依赖）
export DEEPSEEK_API_KEY=***

# 拉取行情数据
python data/fetcher.py --all

# 生成信号
python strategy/signals.py

# 启动完整调度器（推荐）
python scheduler.py

# 启动独立盘中监控
python monitor/monitor_daemon.py
```

---

*最后更新: 2026-05-10*
