# 量化助手 — 面试深度解析

> 用这份文档准备好回答关于这个项目的任何技术问题。
> 重点不是"做了什么"，而是"为什么这么做"和"背后的机制是什么"。

---

## 一、项目定位与面试话术

### 一句话概括
> "一个面向 A 股的量化交易辅助系统，自动化完成行情获取、技术因子计算、AI 信号生成、情绪分析和策略参数调优的全流程，并提供实时可视化仪表盘。"

### 你的角色
说清楚哪些是你独立完成的，哪些是借鉴的：
- **独立完成**：架构设计、FastAPI 路由层、数据管道、前端仪表盘、情绪指数模型
- **借鉴/学习**：技术因子公式（MACD/RSI 等公开算法）、DeepSeek API 集成

### 面试中怎么介绍
1. **首先说痛点**：个人量化交易需要每天手动查数据、算指标、做决策，效率低且容易遗漏信息
2. **然后说方案**：用 Python 全自动拉取行情 → 计算因子 → AI 决策 → 可视化展示
3. **最后说亮点**：30 分钟缓存的情绪指数、FastAPI + Gradio 双端架构、自动参数调优

---

## 二、架构设计（最常被问）

### 2.1 为什么用 Gradio + FastAPI 双架构？

**Gradio 的用途**：提供 AI 对话界面（策略助手对话），利用 Gradio 内置的 Chatbot 组件和 DeepSeek 集成。

**FastAPI 的用途**：提供 RESTful API（`/api/signals`、`/api/status` 等），给前端仪表盘（`frontend.html`）调数据。

**为什么需要两套**？

```
传统做法：全用 Gradio → fn_index 不稳定，自定义 UI 受限
    全用 FastAPI → 需要自己写聊天 UI

本项目做法：FastAPI 提供数据 API，Gradio 的 Chatbot 做对话，两者合并在同一个 App 里
```

关键代码（`app.py` `__main__`）：
```python
from gradio.routes import App
custom_app = App()           # 这是一个 FastAPI 子类
custom_app.add_middleware(...)  # 加 CORS
# 注册 REST 路由
@custom_app.get("/api/signals")
async def api_signals(): ...

app_instance = build_ui()     # 构建 Gradio Blocks
app_instance.launch(_app=custom_app)  # 挂载到同一个 FastAPI 上
```

> **面试点**：Gradio 6.x 的 `App` 类是 FastAPI 子类，`_app` 参数可以传入已有的 FastAPI 实例，实现路由合并。这是 Gradio 6.x 才有的能力。

### 2.2 为什么用 SQLite 不是 MySQL/PostgreSQL？

- 个人项目，不需要并发写入
- 一份文件即可部署，零运维
- Python 自带支持，无需额外依赖
- 数据量级（几万条）SQLite 完全够用

**缺点**：不支持并发写入。解决方案：读写分离——写入用专用连接，读取用另一连接。本项目所有 API 都是读为主。

### 2.3 为什么用 `Promise.allSettled` 不是 `Promise.all`？

```javascript
// 坏：一个接口失败，全部不渲染
const [a, b] = await Promise.all([...])

// 好：单个失败不影响其他
const results = await Promise.allSettled([...])
  .then(r => r.map(x => x.status === 'fulfilled' ? x.value : null));
```

> **面试点**：体现前端容错设计意识。

---

## 三、数据管道详解

### 3.1 完整数据流

```
AKShare (东方财富/新浪)
    │
    ▼
data/fetcher.py ──── 定时或按需拉取
    │
    ├── market_data 表      ← 日线 OHLCV
    ├── northbound_flow 表  ← 北向资金（历史数据）
    ├── margin_data 表      ← 融资融券
    ├── futures_basis 表    ← IF 期货基差
    │
    ▼
strategy/signals.py ──── 技术因子计算
    │
    ├── MACD / RSI / 布林带 / 量价 / KDJ / OBV
    ├── 加权综合评分 (0-100)
    ├── 调用 DeepSeek AI → 方向/仓位/止损
    │
    ▼
signals 表 ──── 存储每日信号
    │
    ▼
storage/db.py ──── 统一数据访问层
    │
    ▼
FastAPI (/api/*) ──── 向外暴露数据
    │
    ▼
frontend.html (Chart.js) ──── 可视化
```

### 3.2 AKShare 是什么？会不会被封？

AKShare 是一个开源的 Python 金融数据库，从东方财富、新浪财经等网站的公开页面抓取数据。本质是**网页爬虫封装**，不是官方 API。

风险：
- 接口随时可能因网页改版而失效（本项目的北向资金接口就停了）
- 频繁请求会被限流（本项目中用 `time.sleep(1.0)` 控制频率）
- 不能用于生产级高频场景

**面试回答**："AKShare 适合个人研究和原型验证，生产环境需要接入券商或万得等收费数据源。"

### 3.3 增量更新机制

```python
def fetch_symbol(symbol):
    latest = get_latest_db_date(symbol)  # 查 DB 中最新日期
    if latest:
        start_date = latest + 1 day      # 只拉取后续数据
    else:
        start_date = 365 days ago        # 首次拉取一年数据
    # 拉取 start_date ~ today
    df = fetch_from_akshare(start_date, today)
    # INSERT OR IGNORE（已存在的不覆盖）
    for _, row in df.iterrows():
        conn.execute("INSERT OR IGNORE INTO market_data ...")
```

优点：每天只需拉取最新 1-2 天的数据，几秒完成。

---

## 四、核心技术因子详解（面试高频）

### 4.1 MACD（40% 权重）

```python
def _calc_macd(close):
    ema_fast = close.ewm(span=12).mean()   # 12日 EMA
    ema_slow = close.ewm(span=26).mean()   # 26日 EMA
    dif = ema_fast - ema_slow              # 快慢线差
    dea = dif.ewm(span=9).mean()           # 信号线
    hist = (dif - dea) * 2                 # 柱状线
```

**面试点**：
- `ewm(span=N).mean()` 是指数加权移动平均，权重按指数衰减，越近的数据权重越大
- `span=12` 意味着半衰期约为 `12 / ln(2) ≈ 8.3` 天
- 金叉（dif 上穿 dea）是看涨信号，死叉是看跌信号

**本项目打分逻辑**：
```python
if hist > 0: macd_score = 60 + min(40, abs(hist) * 2000)  # 柱为正 → 看涨
if 金叉: macd_score = min(100, macd_score + 15)

if hist < 0: macd_score = 40 - min(40, abs(hist) * 2000)  # 柱为负 → 看跌
if 死叉: macd_score = max(0, macd_score - 15)
```

### 4.2 RSI（30% 权重）

```python
def _calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)            # 只保留上涨
    loss = (-delta).clip(lower=0)         # 只保留下跌
    avg_g = gain.ewm(com=period-1).mean() # 平均涨幅
    avg_l = loss.ewm(com=period-1).mean() # 平均跌幅
    rs = avg_g / avg_l
    rsi = 100 - 100 / (1 + rs)
```

**核心逻辑**：RSI > 70 超买（可能回调），RSI < 30 超卖（可能反弹）。

**本项目打分**：
```
RSI ≤ 30 (超卖) → 75-100 分（看涨）
RSI ≥ 70 (超买) → 0-25 分（看跌）
30 < RSI < 70    → 30-70 分（中性，线性映射）
```

### 4.3 布林带（补正因子）

```python
def _calc_bollinger(close, period=20, std_dev=2.0):
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    position = (close - lower) / (upper - lower)
```

**思想**：价格应该在上下轨之间波动。触下轨（position < 0.1）→ 超卖 +5 分；触上轨（position > 0.9）→ 超买 -5 分。

### 4.4 综合决策

```
score = MACD×40% + RSI×30% + 量价×30% + 布林补正

≥ 63 → 买入，仓位 = 0.3 + (score-60)/40 × max_position
≤ 37 → 卖出，仓位同上
38-62 → 观望，仓位 0
```

这个 63 和 37 的阈值怎么来的？—— 经验值。60 分中性偏多，63=60+5% 容错区间。

---

## 五、市场情绪指数模型（独特亮点）

### 5.1 五维度加权模型

| 维度 | 权重 | 信号含义 |
|------|------|---------|
| 融资情绪 | 30% | 融资余额 10 日变化率，>+5% 杠杆资金入场 |
| 主力资金 | 25% | 当日主力净流入，大资金方向 |
| 期货基差 | 20% | IF 近 5 日平均基差，>+0.5% 机构看多 |
| 市场宽度 | 15% | 上证近 5 日上涨天数占比 |
| 指数动量 | 10% | 价格偏离 MA20 百分比 |

### 5.2 基差计算（最有技术含量的维度）

```python
# IF0 = IF 主力连续合约（自动换月）
futures = ak.futures_zh_daily_sina(symbol="IF0")
spot = ak.stock_zh_index_daily_em(symbol="sh000300")
basis = futures.close - spot.close         # 基差 = 期货价 - 现货价
basis_pct = basis / spot.close * 100       # 基差百分比

# 基差 > 0 → 升水（contango），机构看多
# 基差 < 0 → 贴水（backwardation），机构看空/对冲
```

**面试价值**：这展示了金融衍生品知识，不是简单的 HTTP 请求。

### 5.3 缓存机制

```python
_sentiment_cache = {"data": None, "ts": 0}
_SENTIMENT_TTL = 1800  # 30 秒

def fetch_fear_greed_index(config=None):
    now = datetime.now().timestamp()
    
    # 检查缓存
    if _sentiment_cache["data"] and (now - _sentiment_cache["ts"]) < _SENTIMENT_TTL:
        return _sentiment_cache["data"]
    
    # 缓存过期，重新计算
    result = _calc_and_save_sentiment(config)
    _sentiment_cache["data"] = result
    _sentiment_cache["ts"] = now
    return result
```

为什么 TTL 设 30 分钟？—— 交易日 4 小时，30 分钟刷新一次足够。而且 AKShare 接口有可能被封，减少调用频率。

---

## 六、AI 集成（面试高频）

### 6.1 三个 AI 调用点

| 位置 | 用途 | Prompt 策略 |
|------|------|-------------|
| `strategy/signals.py` | 每日信号生成 | 输入技术因子 → 输出 JSON `{direction, confidence, position}` |
| `strategy/ai_engine.py` | 市场分析 | 输入技术信号 + 情绪 → 输出策略建议 |
| `strategy/optimizer.py` | 参数调优 | 输入历史绩效 → 输出参数修改建议 |

### 6.2 Prompt 设计原则

1. **限制输出格式**：`response_format={"type": "json_object"}` 或明确要求 "只输出 JSON"
2. **兜底逻辑**：AI 调用失败时用规则引擎 (`_fallback_signal`) 替代
3. **Temperature 控制**：信号生成用 0.2（低随机性），对话用 0.5

```python
# 容错设计：AI 挂了也不崩
try:
    ai_result = _call_deepseek(prompt)
except:
    ai_result = None

if ai_result is None:
    ai_result = _fallback_signal(score, config)  # 纯规则兜底
```

---

## 七、数据库设计

### 7.1 为什么统一到 `storage/db.py`？

重构前的问题：
- `app.py` 自己写 SQL 查询，`storage/db.py` 也有查询，两套代码
- `storage/db.py` 的 DB 路径是 `data/quant.db`，其他模块用 `db/quant.db` → 两个不同的数据库文件！
- `storage/db.py` 的信号表用 JSON 存储，`strategy/signals.py` 用 flat 结构

重构后：所有模块通过 `storage/db.py` 访问同一个 DB，路径统一，表结构统一。

### 7.2 signals 表为什么要 `ON CONFLICT DO UPDATE`？

```sql
INSERT INTO signals (...) VALUES (...)
ON CONFLICT(symbol, signal_date) DO UPDATE SET ...
```

每天对每个标的生产一条信号。如果调度器在一天内运行多次（重试），后一次覆盖前一次，不会产生重复记录。

---

## 八、前端架构

### 8.1 为什么不用 React/Vue？

项目定位是个人工具，不是面向用户的产品。目标：
- **零构建**：单 HTML 文件，打开即用
- **零依赖**：CDN 加载 Chart.js
- **静态托管**：通过 FastAPI 直接 serve，不需要 Nginx

### 8.2 数据加载策略

```javascript
const API = window.location.protocol.startsWith('file')
  ? 'http://localhost:7860'     // 直接打开文件 → 跨源请求
  : window.location.origin;      // 通过服务器访问 → 同源

// 6 个接口并行请求，任何一个失败不影响其他
const [signals, perf, cfg, equity, sent] = await Promise.allSettled([
  apiGet('/api/signals'), ...
]).then(results => results.map(r => r.status === 'fulfilled' ? r.value : null));
```

### 8.3 Mock 数据策略

```javascript
// 种子随机数：同一天内结果一致，不会刷新就变
function seededRandom(seed) {
  let s = seed;
  return function() {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}
```

`16807` 是 Lehmer 随机数生成器的标准乘数。

---

## 九、安全设计

| 风险 | 防护措施 |
|------|---------|
| API Key 泄露 | 全部通过环境变量读取，无硬编码 |
| 数据库泄露 | `.gitignore` 忽略 `db/` 目录 |
| XSS 攻击 | FastAPI 自动 HTML 转义 |
| CORS 跨域 | `strict_cors=False` + `allow_origins=["*"]`（本地项目可接受） |

---

## 十、面试可能问的问题（附参考回答）

### Q1: 这个项目最大的技术挑战是什么？

> **参考回答**：最大的挑战是 Gradio 6.x 的自定义路由问题。Gradio 的新版本不再允许直接往 `app.app` 上添加 FastAPI 路由，这些路由会在 `launch()` 时被丢弃。查了两天源码才发现需要用 `gradio.routes.App` 创建实例，通过 `_app=custom_app` 参数传入 `launch()` 方法，才能让自定义路由和 Gradio UI 共存。

### Q2: 如果数据源挂了怎么办？

> **参考回答**：我们有三级容错：
> 1. **API 调用失败** → 重试 3 次，间隔递增（2s, 4s, 6s）
> 2. **AI 决策失败** → 回退到纯规则引擎（综合评分 > 63 就买入）
> 3. **前端请求失败** → 显示空状态 + 提示"暂无数据"，不崩

### Q3: 这个策略能赚钱吗？

> **参考回答**：这个项目的定位是辅助决策工具，不是自动交易机器人。它的价值在于：
> - 把 AI 的分析能力和人的判断结合起来（人做最终决策）
> - 自动化了每天的数据收集和因子计算工作
> - 参数调优模块提供了量化分析的方向性参考
> 实盘需要考虑滑点、手续费、冲击成本等，这些当前版本还没覆盖。

### Q4: 如果数据量变大（百万级）怎么办？

> **参考回答**：当前 SQLite 可以支撑百万级数据。如果继续增长，可以：
> 1. 加索引（当前没有索引，按 date 查询可优化）
> 2. 分区存储（按年份分表）
> 3. 升级到 PostgreSQL（需要改 `storage/db.py` 的 SQL 方言）

### Q5: 情绪指数怎么验证准确度？

> **参考回答**：当前没有回测验证。理想做法是：
> - 情绪指数作为信号 vs 次日大盘涨跌做相关性分析
> - 如果情绪 < 45（恐惧）时次日上涨概率 > 60%，说明指标有效
> - 这需要一个回测框架来验证，是后续改进方向

---

## 十一、文件地图（方便面试时快速定位）

```
app.py                    # 应用入口，FastAPI 路由 + Gradio UI
├── build_ui()            # 构建 Gradio 界面
├── chat()                # DeepSeek 对话函数
├── get_client()          # API 客户端初始化
├── api_get_*_json()      # 6 个数据 API 函数
└── __main__              # 启动逻辑（路由注册 + 运行）

data/fetcher.py           # 数据拉取层
├── fetch_symbol()        # 单标的行情拉取（增量）
├── update_all()          # 批量拉取
├── get_ohlcv()           # 查行情数据
├── fetch_northbound_flow()  # 北向资金
├── fetch_margin_data()   # 融资融券
├── fetch_futures_basis() # 期货基差
├── fetch_fear_greed_index()  # 情绪指数（带缓存）
└── _calc_sentiment_score()   # 情绪计算核心

strategy/signals.py       # 技术因子 + 信号生成
├── generate_signal()     # 单标的信号生成
├── _calc_macd/rsi/bollinger/etc  # 因子计算
├── _composite_score()    # 综合评分
├── calculate_all_signals()  # scheduler 适配
└── run_backtest()        # 历史回测

strategy/ai_engine.py     # DeepSeek AI 决策
├── analyze_market()      # 市场分析
├── nightly_optimize()    # 夜间调优
└── _parse_json_response()  # AI 输出解析

strategy/optimizer.py     # 参数调优
├── run_nightly_optimization()
├── validate_changes()    # 参数边界校验
└── backup_config()/rollback_

storage/db.py             # 统一数据访问层
├── get_latest_signals()
├── get_performance_summary()
├── get_equity_curve()
└── get_db_status()       # 系统监控

frontend.html             # 前端仪表盘（单文件）
├── loadAll()             # 数据加载
├── drawEquityChart()     # 净值曲线
├── drawSentimentGauge()  # 情绪仪表盘
├── sendMessage()         # 对话
└── refreshStatus()       # 状态栏

scheduler.py              # APScheduler 定时任务
config.yaml               # 策略配置
```

---

*生成时间: 2026-05-03*
