# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

**Video-Captions** - 视频字幕下载工具

- 支持 B站、YouTube 和本地文件
- 从 API 直接获取字幕，无字幕时使用 Whisper ASR 自动生成
- 自动繁简转换
- 提供 CLI 和 MCP 两种使用方式
- 支持自动从浏览器读取 Cookie（Chrome、Edge、Firefox、Brave）

**Python 版本：** >=3.10
**系统依赖：** yt-dlp, ffmpeg（`brew install yt-dlp ffmpeg`）
**ASR 支持：** 仅限 Apple Silicon (M1/M2/M3/M4) Mac

## 架构设计

三层架构，职责清晰：

```
src/
├── handler/           # 接入层：CLI (cli.py) 和 MCP (mcp.py) 入口
├── service/           # 业务层：平台服务实现，继承 SubtitleService 抽象基类
│   ├── base.py        # 抽象基类，定义所有服务必须实现的接口
│   ├── bilibili.py    # B站服务
│   ├── youtube.py     # YouTube 服务
│   └── local.py       # 本地文件服务
└── core/              # 基础层：通用功能
    ├── asr.py         # Whisper ASR 转录
    ├── audio.py       # 音频提取（ffmpeg）
    ├── browser.py     # 浏览器 Cookie 读取
    ├── cookie.py      # Cookie 管理
    ├── formatter.py   # 字幕格式化（text/srt/json）
    ├── text.py        # 文本处理（繁简转换）
    └── logging.py     # 日志系统
```

**扩展新平台：** 在 `service/` 下创建新类继承 `SubtitleService`，实现 `is_supported()`, `get_info()`, `list_subtitles()`, `download_subtitle()` 等方法。然后在 `service/__init__.py` 的 `_SERVICE_REGISTRY` 和 `get_service()` 中注册，匹配优先级为 local > bilibili > youtube。

**服务工厂：** `service/__init__.py` 提供 `get_service(source, browser)` 根据来源自动识别并返回对应服务实例。`LocalService` 不需要 browser 参数，`BilibiliService`/`YouTubeService` 需要。

## MCP 工具

MCP 服务器（`handler/mcp.py`）暴露两个 tool：
- `download_captions` — 在线视频字幕下载（B站/YouTube），参数：url, format, model_size, browser
- `transcribe_local_file` — 本地文件 ASR 转录，参数：file_path, format, model_size

## 开发命令

### 环境设置
```bash
uv sync              # 安装依赖
uv sync --dev        # 安装开发依赖（pytest, ruff）
```

### 运行和测试
```bash
# CLI 运行
uv run video-captions <URL>
uv run video-captions --browser chrome --format srt --model small <URL>

# MCP 服务器
uv run video-captions-mcp

# 测试
uv run pytest tests/                    # 运行所有测试
uv run python tests/test_bilibili.py    # 单独测试 B站
uv run python tests/test_youtube.py     # 单独测试 YouTube
make test                                # 或使用 Makefile
```

### 代码质量
```bash
uv run ruff check .     # 代码检查
uv run ruff format .    # 代码格式化
make lint               # 或使用 Makefile
make format             # 或使用 Makefile
```

**Ruff 配置：** line-length=100, target-version=py310

### 构建和发布
```bash
uv build                # 构建包（输出到 dist/）
make build              # 或使用 Makefile
make clean              # 清理构建文件
```

**版本管理：** 推送到 main 分支后，GitHub Actions 自动生成版本号 `0.1.YYYYMMDD.序号` 并发布到 PyPI。无需手动管理版本号。

## 包结构

**入口点（pyproject.toml）：**
- `video-captions` → `handler.cli:main`
- `video-captions-mcp` → `handler.mcp:main`

**包映射：** `src/` 下的 `core/`, `handler/`, `service/` 三个目录会被打包到根命名空间。

## CLI 选项

| 选项 | 说明 |
|------|------|
| `--browser` | 从浏览器读取 Cookie: `auto`(默认) / `chrome` / `edge` / `firefox` / `brave` |
| `--model` | ASR 模型: `base` / `small` / `medium` / `large`(默认) |
| `--format` | 输出格式: `text`(默认) / `srt` / `json` |
| `--verbose, -v` | 显示详细日志 |

## 关键依赖

**系统依赖：**
- **yt-dlp**: 下载视频（`brew install yt-dlp`）
- **ffmpeg**: 提取音频（`brew install ffmpeg`）

**Python 依赖：**
- **mlx-whisper** (>=0.4.0): ASR 语音识别（Apple Silicon 优化）
- **browser-cookie3** (>=0.19.0): 从浏览器读取加密 Cookie
- **opencc-python-reimplemented** (>=0.1.7): 繁简转换
- **mcp** (>=1.0.0): MCP 协议支持

## 开发注意事项

- 调用 yt-dlp 时必须使用 `--quiet` 参数（仅 `--no-progress` 不够，无法抑制 "Download complete" 等消息）`[added: 2026-04-05]`
- 抑制第三方库输出时，`subprocess.run(capture_output=True)` 只能抓子进程、`TQDM_DISABLE=1` 只能禁 tqdm 进度条、`contextlib.redirect_stdout` 只能抓 Python 级输出。huggingface_hub 的 "Download complete" 是 C 扩展直接写 fd，必须用 `os.dup2(devnull, 1/2)` 在 OS fd 层面重定向才可靠 `[updated: 2026-04-05]`
- CLI 非 verbose 模式下 stdout 只能输出字幕内容，所有进度/日志信息必须走 stderr 且受 `_verbose_log` 控制 `[added: 2026-04-05]`

## 测试视频

| 视频 | 平台 | 场景 |
|------|------|------|
| BV16YC3BrEDz | B站 | 有 API 字幕 |
| BV1qViQBwELr | B站 | 无字幕（ASR 兜底） |
| kQ-aFczITCg | YouTube | 有 API 字幕 |
| 5GJU5-UMNWk | YouTube | 无字幕（ASR 兜底） |
