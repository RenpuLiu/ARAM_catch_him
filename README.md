# ARAM Catch Him

本地版英雄联盟 ARAM 战绩分析小工具。它只读取你当前登录的 League Client / LCU 本机接口，把最近对局保存到本地，并生成 CSV、SQLite、Streamlit 看板和可选的 LLM 复盘报告。

## TLDR：最速运行指南

先打开并登录英雄联盟客户端，然后在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python lol_cn_lcu_dump.py --n 50 --out data
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

看板里可以点“从 LCU 更新”继续刷新数据。需要 LLM 复盘时，把 `.env.example` 复制成 `.env` 并填入 `OPENAI_API_KEY`，然后在看板的 `LLM 分析` tab 生成报告。

## 功能概览

- 自动寻找 League Client `lockfile`，也支持手动传入。
- 拉取当前召唤师、最近 N 局、单局详情，可选 timeline。
- 累积保存 `matches.csv`、`participants.csv` 和 `matches.sqlite`，按对局去重合并历史。
- Streamlit 看板展示胜率、近况、英雄池、队友/对手、装备、阵容复盘等。
- 可下载 Data Dragon `zh_CN` 静态数据，显示中文英雄、装备和召唤师技能名。
- 可选 LLM 分析：基于本地统计生成车队复盘、常见搭配、最近表现排序和改进建议。

## 常用命令

抓取最近 50 局：

```powershell
python lol_cn_lcu_dump.py --n 50 --out data
```

抓取 timeline，速度会慢一些：

```powershell
python lol_cn_lcu_dump.py --n 50 --out data --timeline
```

自动找不到 `lockfile` 时手动指定：

```powershell
python lol_cn_lcu_dump.py --n 50 --out data --lockfile "C:\path\to\League of Legends\lockfile"
```

启动看板：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

端口被占用时换一个：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

停止看板：在启动它的 PowerShell 窗口按 `Ctrl+C`。

## LLM 分析

推荐用 `.env` 保存配置，`.env` 已被 `.gitignore` 忽略。最少只需要：

```text
OPENAI_API_KEY=sk-your-key-here
```

其他模型、网关、超时、输出长度和车队名单配置见 `.env.example`。

也可以直接用命令行生成报告：

```powershell
.\.venv\Scripts\python.exe llm_analyze.py --data data --recent-games 50 --min-partner-games 2
```

只预览上下文、不调用 API：

```powershell
.\.venv\Scripts\python.exe llm_analyze.py --dry-run --data data
```

报告会保存到：

```text
data/reports/
```

如果报告被截断，把 `LLM_MAX_OUTPUT_TOKENS` 调高；如果看到 `ReadTimeout`，把 `LLM_TIMEOUT_SECONDS` 调到 `600` 或 `900`。网关只兼容 Chat Completions 时设置 `LLM_API_STYLE=chat`。

## 分析 Skill

默认系统 prompt 在：

```text
skills/aram-match-analyst/SKILL.md
```

自定义 skill 可以放到：

```text
skills/user/<your-skill-name>/SKILL.md
```

看板会自动发现 `skills/user/*/SKILL.md`。命令行可以指定一个或多个 skill：

```powershell
.\.venv\Scripts\python.exe llm_analyze.py `
  --skill skills/aram-match-analyst/SKILL.md `
  --skill skills/user/my-extra-style/SKILL.md
```

默认 skill 会把分析对象视为整个车队，对 `players_for_equal_analysis` 中的成员做同等深度分析。可用 `.env` 里的 `LLM_SQUAD_MEMBERS` 调整车队名单。

## 输出目录

```text
data/
  raw/                 原始 LCU JSON 和单局归档
  cache/               Data Dragon 中文静态映射
  reports/             LLM 分析报告
  matches.csv          累积对局表
  participants.csv     累积玩家表现表
  matches.sqlite       SQLite 数据库
  summary_basic.json   基础统计摘要
```

`matches.csv` 和 `participants.csv` 是累积历史，不只是本次抓取结果。重复对局会按 `match_id` 去重，SQLite 会用合并后的完整历史重建。

## 注意

这个项目只读本机客户端中你自己的账号数据，不走 WeGame/掌盟私有接口逆向。LCU API 不是稳定公开服务，不同客户端版本字段可能略有差异，所以 raw JSON 会完整保留，方便后续调整 parser。

国服 ARAM 可能不是标准 `queueId=450`，而是 `queueId=2400`、`gameMode=KIWI`、`mapId=12`。当前版本会把这些形态都识别为 ARAM。
