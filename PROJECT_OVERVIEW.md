# 量化助手 — 项目概览

A股量化交易助手，Gradio + FastAPI 双前端界面，DeepSeek AI 驱动。自动化数据拉取、技术因子计算、信号生成、参数调优、情绪分析全流程。

---

## 架构总览

```
                       ┌─────────────────────┐
                       │   frontend.html      │  ← 自定义仪表盘 (Chart.js)
                       │  http://:7860/frontend│
                       └─────┬───────────────┘
                             │ GET/POST
                       ┌─────▼───────────────┐
                       │   FastAPI (custom)   │  ← /api/* 路由
                       │   gradio.routes.App  │
                       └─────┬───────────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
         ┌─────▼──┐   ┌─────▼──┐   ┌─────▼──┐
         │ fetcher│   │ signals│   │ai_engine│
         │ .py    │   │ .py    │   │ .py     │
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
         ┌───▼────────────▼────────────▼───┐
         │        SQLite (quant.db)        │
         │ market_data / signals /         │
         │ northbound_flow / sentiment_    │
         │ index / config_versions         │
         └────────────────────────────────┘
```

### 启动入口

```bash
python app.py
# → 仪表盘: http://127.0.0.1:7860/frontend
# → Gradio: http://127.0.0.1:7860
```

`app.py` 创建 `gradio.routes.App` 实例，注册所有 `/api/*` FastAPI 路由后，将 Gradio UI 挂载到该自定义 App 上启动。

---

## 工作流程

### 每日自动循环（由 scheduler.py + APScheduler 驱动）

```
09:00 ── data/fetcher.update_all()
           ├── AKShare 拉取行情 (OHLCV)
           ├── INSERT OR IGNORE → market_data 表
           └── 增量更新（跳过已有日期）

09:10 ── data/fetcher.fetch_fear_greed_index()
           ├── 计算多维度情绪指数 0-100
           └── INSERT OR REPLACE → sentiment_index 表

09:15 ── strategy/signals.run_all()
           ├── 读 market_data → 计算技术因子
           ├── 调用 DeepSeek AI 决策
           └── INSERT → signals 表

22:00 ── strategy/optimizer.run_nightly_optimization()
           ├── 读近期 signals 绩效
           ├── 调 DeepSeek 建议参数变更
           ├── 备份旧配置 → 写入新 config.yaml
           └── 记录 config_versions 表
```

### 前端数据流

```
用户打开 /frontend
  → loadAll() 并行 GET /api/* (Promise.allSettled)
  → 各 endpoint 查 DB，无数据时自动触发拉取
  → Chart.js 渲染 5 张图表/指标
  → 每 5 分钟自动刷新
```

---

## 数据源

| 数据 | 来源 | 存储表 | 更新频率 | 说明 |
|------|------|--------|----------|------|
| A股日线行情 | AKShare (东方财富) | `market_data` | 每交易日 | 前复权，支持 index/etf/stock |
| 北向资金 | AKShare (数据截至 2024-08) | `northbound_flow` | 手动 | 沪港通/深港通净流入历史 |
| 融资融券 | AKShare 沪深交易所 | `margin_data` | 每交易日 | 融资余额变化，杠杆资金态度 |
| 股指期货升贴水 | AKShare Sina 期货 | `futures_basis` | 每交易日 | IF 主力基差，机构情绪 |
| 市场情绪 | 自行计算 | `sentiment_index` | 每交易日 | 5 维度加权合成 |
| 交易信号 | DeepSeek AI 决策 | `signals` | 每交易日 | 含方向/置信度/仓位/止损方向/置信度/仓位/止损 |
| 策略配置 | `config.yaml` | `config_versions` | 调优时 | 版本化管理，支持回滚 |

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

≥ 63 → buy,  仓位 = 0.3 + (score-60)/40×max_position
≤ 37 → sell, 仓位同上
38-62 → hold, 仓位 0
```

### KDJ / OBV（辅助参考）

KDJ、OBV 作为辅助指标参与信号生成，不直接计入评分权重。

---

## 市场情绪指数（data/fetcher.py）

### 5 维度加权合成 (0-100)

| 维度 | 权重 | 数据来源 |
|------|------|----------|
| 融资情绪 | 30% | `margin_data` 表，近 10 日融资余额变化率 |
| 主力资金流向 | 25% | AKShare `stock_market_fund_flow()` |
| 期货升贴水 | 20% | `futures_basis` 表，IF 主力近 5 日平均基差 |
| 市场宽度 | 15% | 上证指数近 5 日上涨天数占比 |
| 指数动量 | 10% | 指数价格偏离 MA20 的百分比 |

### 标签区间
```
extreme_fear  (0-25) → 🔴 极度恐惧
fear          (25-45) → 🟠 恐惧
neutral       (45-55) → ⚪ 中性
greed         (55-75) → 🟢 贪婪
extreme_greed (75-100) → 💚 极度贪婪
```

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

### futures_basis（期货升贴水）
| 字段 | 类型 | 说明 |
|------|------|------|
| date | TEXT | 交易日 (PRIMARY KEY) |
| futures_close | REAL | IF 主力合约收盘价 |
| spot_close | REAL | 沪深 300 现货收盘价 |
| basis | REAL | 升贴水 = 期货 - 现货 |
| basis_pct | REAL | 升贴水百分比 |

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

夜间 22:00 自动执行：
1. 查询最近 10 天 signals 绩效（胜率/平均收益/方向分布）
2. 构建 Prompt 发给 DeepSeek
3. DeepSeek 返回参数调整建议（R SI 阈值/周期/权重/止损）
4. 参数边界校验（clamp 到合法范围）
5. 备份旧 `config.yaml` → 写入新配置
6. 支持 `--dry-run` 预览不写入

### 调优原则
- 胜率 < 45% → 优先收紧止损，降低仓位上限
- 平均收益 < 0% → 调整 RSI 超买超卖阈值
- 胜率 > 65% 且收益 > 0 → 适当放宽仓位上限
- 单次参数变动 ≤ 当前值 20%（防止跳变）

---

## 前端仪表盘

5 个 KPI 指标：
- 累计净值 | 近30日胜率 | 平均单次收益 | 最大回撤 | 市场情绪

5 个图表/组件：
- **净值 & 回撤曲线** — 近 60 日累计净值 + 最大回撤（双轴）
- **信号方向分布** — 饼图，买入/卖出/观望占比
- **市场情绪仪表盘** — 0-100 环形图，含维度分解
- **最新信号卡片** — 今日信号，方向/置信度/仓位/收益
- **信号历史表格** — 完整信号记录，带收益结算状态

支持深色/浅色主题切换，5 分钟自动刷新。

---

## 快速开始

```bash
# 安装依赖
pip install akshare gradio openai pyyaml plotly pandas "httpx<0.28"

# 设置 API Key（DeepSeek）
export DEEPSEEK_API_KEY=sk-xxxx

# 拉取行情数据
python data/fetcher.py

# 计算情绪指数
python data/fetcher.py --sentiment

# 生成信号
python strategy/signals.py

# 启动 Web 界面
python app.py
# → 仪表盘: http://127.0.0.1:7860/frontend
# → Gradio: http://127.0.0.1:7860
```

---

*生成时间: 2026-05-03*
