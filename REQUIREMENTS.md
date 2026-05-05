# 量化助手 — 依赖清单

## 必需依赖

```txt
gradio>=6.0
openai>=1.0
akshare>=1.14
pandas>=2.0
pyyaml>=6.0
plotly>=5.0
apscheduler>=3.10
httpx<0.28
requests>=2.0
```

## 一键安装

```bash
pip install gradio openai akshare pandas pyyaml plotly apscheduler "httpx<0.28" requests
```

## 各依赖用途

| 包名 | 用途 |
|------|------|
| `gradio` | Web UI 框架（对话界面 + 服务器） |
| `openai` | 调用 DeepSeek API（兼容 OpenAI SDK） |
| `akshare` | 拉取 A 股行情、融资融券、期货等金融数据 |
| `pandas` | 数据处理、因子计算、数据库读写 |
| `pyyaml` | 读写 `config.yaml` 策略配置文件 |
| `plotly` | Gradio 中的净值曲线图表 |
| `apscheduler` | 定时调度（每日行情拉取、信号生成、夜间调优） |
| `httpx` | Gradio 6.x 内部依赖，锁定 `<0.28` 避免版本兼容问题 |
| `requests` | DeepSeek API 调用（ai_engine.py 用） |

## 可选依赖

```bash
pip install numpy  # akshare 的间接依赖，通常自动安装
```

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥，用于 AI 信号生成和对话 |

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-你的key"

# Windows CMD
set DEEPSEEK_API_KEY=sk-你的key

# Linux / Mac
export DEEPSEEK_API_KEY=sk-你的key
```

## 验证安装

```bash
python -c "
import gradio
import openai
import akshare
import pandas
import yaml
import plotly
import apscheduler
import requests
print('所有依赖安装成功')
"
```
