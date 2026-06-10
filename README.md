# ARAM Catch Him

一个本地版英雄联盟 ARAM 战绩分析小工具。它只读取你当前登录的 League Client / LCU 本机接口，把自己的最近对局保存到本地，再生成 CSV、SQLite 和 Streamlit 看板。

## 功能

- 自动寻找 League Client `lockfile`
- 拉取当前召唤师、最近 N 局、单局详情，可选 timeline
- 保存 raw JSON 到 `data/raw/`
- 规范化对局字段到 `data/matches.csv` 和 `data/matches.sqlite`
- 展开每局 10 个玩家到 `data/participants.csv` 和 SQLite `participants` 表
- 统计 ARAM 胜率、近 20 局、KDA、英雄池、时间段、装备出现次数
- 看板里查看自己的最近对局、队友、对手和按局分组的阵容复盘
- 阵容视图按每局拆分，默认展示两边队伍的紧凑表；详细字段里包含装备、召唤师技能、符文/增强 id、伤害拆分、治疗、控制、视野、金币等
- LLM 分析：基于本地对局统计生成用户表现、常见队友/疑似多排、搭配表现、游戏风格、最近表现排序和改进建议报告
- 可选下载 Data Dragon `zh_CN` 静态数据，显示中文英雄名和装备名

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 抓取数据

先打开并登录英雄联盟客户端，然后运行：

```powershell
python lol_cn_lcu_dump.py --n 50 --out data
```

如果自动找不到 `lockfile`：

```powershell
python lol_cn_lcu_dump.py --n 50 --out data --lockfile "C:\path\to\League of Legends\lockfile"
```

如果看到 `Invalid lockfile format` 或 `Lockfile is empty`，通常是国服客户端目录里有一个空的占位 `lockfile`。当前版本会自动跳过无效候选，并从 LeagueClient 进程参数里兜底读取端口和临时 token。保持客户端登录状态后重试即可。

需要 timeline 时：

```powershell
python lol_cn_lcu_dump.py --n 50 --out data --timeline
```

## 打开看板

```powershell
streamlit run app.py
```

看板里也可以直接点“从 LCU 更新”刷新数据。

## LLM 分析

看板里的 `LLM 分析` tab 可以先预览将发送给模型的统计摘要。填写 API Key 后点击“生成 LLM 分析报告”，报告会保存到：

```text
data/reports/
```

你可以直接在项目根目录创建 `.env` 存放 API key。这个文件已经被 `.gitignore` 忽略，不会被提交：

```text
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-5.4
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_STYLE=responses
LLM_MAX_OUTPUT_TOKENS=8000
LLM_REASONING_EFFORT=low
```

项目里有一个模板文件：

```text
.env.example
```

也支持 Streamlit secrets：

```text
.streamlit/secrets.toml
```

也可以用命令行：

```powershell
$env:OPENAI_API_KEY="你的 key"
$env:LLM_MODEL="gpt-5.4"
$env:LLM_MAX_OUTPUT_TOKENS="8000"
.\.venv\Scripts\python.exe llm_analyze.py --data data --recent-games 50 --min-partner-games 2
```

兼容 OpenAI Responses API 的默认配置：

```powershell
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_API_STYLE="responses"
```

如果你的网关只兼容 Chat Completions：

```powershell
$env:LLM_API_STYLE="chat"
```

如果生成出的报告看起来突然停在半句，或者报告开头提示 API 返回 `incomplete` / `max_output_tokens`，说明模型输出被长度上限截断了。把 `LLM_MAX_OUTPUT_TOKENS` 调到 `8000`、`12000` 或更高后重新生成；也可以减少“分析局数”。Responses API 下可设置 `LLM_REASONING_EFFORT=low`，避免隐藏推理占用过多输出预算。

只预览上下文、不调用 API：

```powershell
.\.venv\Scripts\python.exe llm_analyze.py --dry-run --data data
```

## 分析 Skill

默认系统 prompt 是一个项目内 skill：

```text
skills/aram-match-analyst/SKILL.md
```

你可以把自己的分析 skill 放到：

```text
skills/user/<your-skill-name>/SKILL.md
```

每个 `SKILL.md` 使用同样结构：

```markdown
---
name: my-aram-analyst
description: Custom ARAM analysis style
---

# My ARAM Analyst

## System Prompt

你的系统 prompt...
```

看板会自动发现 `skills/user/*/SKILL.md`，命令行也可以指定：

```powershell
.\.venv\Scripts\python.exe llm_analyze.py --skill skills/user/my-aram-analyst/SKILL.md
```

看板里的 `LLM 分析` 支持多选 skill。命令行也可以传多个 `--skill`，它们会按顺序合并成一个 system prompt：

```powershell
.\.venv\Scripts\python.exe llm_analyze.py `
  --skill skills/aram-match-analyst/SKILL.md `
  --skill skills/user/my-extra-style/SKILL.md
```

默认 `aram-match-analyst` 会要求模型对你本人和每个常见队友/疑似多排做同维度分析，并生成最近表现排序。上下文里会包含：

```text
players_for_equal_analysis
recent_performance_ranking_seed
frequent_allies
```

## 输出目录

```text
data/
  raw/
    current_summoner.json
    matchlist_raw.json
    games_from_matchlist.json
    game_details_raw.json
    timelines_raw.json
    fetch_errors.json
  cache/
    ddragon_zh_CN.json
  matches.csv
  participants.csv
  matches.sqlite
  summary_basic.json
  reports/
```

`participants.csv` 每行是一名玩家在一局里的表现。常用字段包括：

```text
match_id, side, team_id, riot_id, champion_id, win
kills, deaths, assists, kda, champ_level
damage_to_champions, physical_damage_to_champions, magic_damage_to_champions, true_damage_to_champions
damage_taken, damage_self_mitigated, total_heal, time_ccing_others
gold_earned, gold_spent, cs
spell1_id, spell2_id, item0...item6, items_json
perks_json, augments_json
```

## 注意

国服个人战绩不要走 WeGame/掌盟私有接口逆向。这个项目只读本机客户端中你自己的账号数据，适合做本地复盘和统计。LCU API 不是稳定公开服务，不同客户端版本的字段可能略有差异，所以 raw JSON 会完整保留，后续 parser 可以继续按你的数据微调。

国服 ARAM 在这套客户端里可能不是标准 `queueId=450`，而是 `queueId=2400`、`gameMode=KIWI`、`mapId=12`。当前版本会把这些形态都识别为 ARAM。
