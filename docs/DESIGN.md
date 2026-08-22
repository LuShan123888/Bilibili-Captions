# video-captions 设计规划文档

## 1. 项目定位

video-captions 是一个视频字幕获取工具，支持从 B站、YouTube 在线视频和本地音频/视频文件中提取字幕内容。优先通过平台 API 直接获取结构化字幕，当 API 无字幕时自动使用 Whisper ASR 语音识别生成字幕。

### 1.1 目标用户

| 用户 | 场景 |
|------|------|
| AI 编码工具用户 | 通过 MCP 接口获取视频字幕，用于内容分析、文章生成等 |
| 命令行开发者 | 通过 CLI 快速获取视频字幕内容 |
| 内容创作者 | 提取视频字幕用于文案整理、翻译参考 |

### 1.2 核心价值

| 价值 | 说明 |
|------|------|
| API 优先 | 优先从平台 API 获取结构化字幕，速度快、质量高 |
| ASR 兜底 | API 无字幕时自动切换 Whisper ASR，无需用户干预 |
| 双入口 | CLI 和 MCP 两种使用方式，覆盖终端和 AI 工具场景 |
| 自动繁简转换 | 输出统一为简体中文，消除繁简混排问题 |
| 多浏览器 Cookie | 自动从 Arc/Chrome/Edge/Firefox/Brave 读取登录态 |

---

## 2. 当前架构

### 2.1 三层架构

```
┌────────────────────────────────────────────┐
│              Handler 层 (接入层)            │
│  ┌──────────────┐      ┌──────────────┐     │
│  │   CLI 入口    │      │   MCP 入口    │     │
│  │  cli.py      │      │  mcp.py      │     │
│  │  argparse    │      │  FastMCP     │     │
│  └──────┬───────┘      └──────┬───────┘     │
└─────────┼─────────────────────┼─────────────┘
          │                     │
          └──────────┬──────────┘
                     │
┌────────────────────┼────────────────────────────────────┐
│                    Service 层 (业务层)                    │
│                     │                                    │
│              ┌──────┴──────┐                             │
│              │  服务工厂     │  service/__init__.py       │
│              │ get_service()│                             │
│              └──────┬──────┘                             │
│                     │                                    │
│     ┌───────────────┼───────────────┐                    │
│     ▼               ▼               ▼                    │
│ ┌─────────┐  ┌──────────┐  ┌──────────┐                │
│ │  Local   │  │ Bilibili │  │ YouTube  │                │
│ │ Service  │  │ Service  │  │ Service  │                │
│ │(ASR only)│  │(API+ASR) │  │(API+ASR) │                │
│ └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│      └──────────────┴────────────┘                       │
│                     │                                    │
│              SubtitleService (抽象基类)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────┐
│                    Core 层 (基础层)                       │
│     ┌───────────────┼───────────────┐                    │
│     ▼               ▼               ▼                    │
│ ┌─────────┐  ┌──────────┐  ┌──────────┐                │
│ │  ASR    │  │  Audio   │  │Formatter │                │
│ │ mlx-    │  │ ffmpeg   │  │text/srt/ │                │
│ │ whisper │  │ 提取音频  │  │ json     │                │
│ └─────────┘  └──────────┘  └──────────┘                │
│ ┌─────────┐  ┌──────────┐  ┌──────────┐                │
│ │ Browser │  │  Cookie  │  │  Text    │                │
│ │ Cookie  │  │  管理    │  │ 繁简转换  │                │
│ │ 读取    │  │          │  │          │                │
│ └─────────┘  └──────────┘  └──────────┘                │
│ ┌──────────┐                                            │
│ │ Logging  │                                            │
│ │ 统一日志  │                                            │
│ └──────────┘                                            │
└─────────────────────────────────────────────────────────┘
```

**关键设计决策**：Handler 层只做参数解析和结果展示，所有业务逻辑在 Service 层实现，通用功能下沉到 Core 层。三个层之间通过明确的接口通信，互不耦合。CLI 通过 `--help` 直接提供完整的命令和参数说明，无需额外的 Agent 封装。

### 2.2 服务工厂与注册表

```python
# 服务注册表
_SERVICE_REGISTRY: Dict[str, Type[SubtitleService]] = {
    "local": LocalService,
    "bilibili": BilibiliService,
    "youtube": YouTubeService,
}
```

**平台识别优先级**：`local` > `bilibili` > `youtube`

识别逻辑：
1. 检查是否为本地文件路径（存在且为音视频格式）→ `LocalService`
2. 匹配 B站 URL 模式（`bilibili.com/video/`、`bilibili.com/list/`、`BV` 号）→ `BilibiliService`
3. 匹配 YouTube URL 模式（`youtube.com/watch`、`youtu.be/`、`youtube.com/shorts/` 等）→ `YouTubeService`

### 2.3 抽象基类接口

`SubtitleService` 定义了所有服务必须实现的接口：

| 方法 | 说明 | 参数 |
|------|------|------|
| `name` | 服务名称属性 | — |
| `is_supported(source)` | 判断是否支持该来源 | URL 或文件路径 |
| `get_info(source)` | 获取视频/文件基本信息 | URL 或文件路径 |
| `list_subtitles(source)` | 列出可用字幕 | URL 或文件路径 |
| `download_subtitle(source, format, model_size)` | 下载字幕（API 优先，ASR 兜底） | URL/路径 + 格式 + 模型大小 |
| `download_video(source, output_dir)` | 下载视频文件 | URL + 输出目录 |
| `extract_audio(video_file, output_dir)` | 从视频提取音频 | 视频文件路径 + 输出目录 |
| `download_and_extract_audio(source, output_dir)` | 下载视频并提取音频（默认实现） | URL + 输出目录 |

### 2.4 字幕获取流程

```
                    ┌──────────────┐
                    │   输入来源    │
                    │  URL/文件路径  │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │  服务工厂     │
                    │ get_service() │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Local   │ │ Bilibili │ │ YouTube  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             │       ┌────┴────┐       │
             │       │ 有字幕？  │       │
             │       └────┬────┘       │
             │       ┌────┴────┐       │
             │    是 │         │ 否     │
             │       ▼         ▼       │
             │  ┌────────┐ ┌──────┐    │
             │  │API 获取 │ │ASR   │    │
             │  │结构化   │ │下载  │    │
             │  │字幕     │ │视频  │    │
             │  └───┬────┘ │提取  │    │
             │      │      │音频  │    │
             │      │      │ASR   │    │
             │      │      │识别  │    │
             │      │      └──┬───┘    │
             │      │         │        │
             └──────┼────┬────┼────────┘
                    │    │    │
                    ▼    ▼    ▼
              ┌─────────────────┐
              │   繁简转换       │
              │ convert_to_     │
              │ simplified()    │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │   格式化输出      │
              │ text/srt/json   │
              └─────────────────┘
```

### 2.5 技术选型

| 维度 | 选型 | 说明 |
|------|------|------|
| 语言 | Python >= 3.10 | ASR 和 MCP 生态 |
| 构建 | hatchling | 轻量、src 布局支持好 |
| 包管理 | uv | 快速、现代 |
| HTTP 客户端 | httpx (异步) | B站 API 调用 |
| ASR 引擎 | mlx-whisper | Apple Silicon 优化 |
| Cookie 读取 | browser-cookie3 | 解密浏览器 Cookie |
| 繁简转换 | opencc-python-reimplemented | 纯 Python 实现 |
| 视频下载 | yt-dlp (subprocess) | 命令行调用，灵活 |
| 音频提取 | ffmpeg (subprocess) | 命令行调用，稳定 |
| MCP 框架 | FastMCP | 官方推荐 |
| 代码质量 | ruff (lint + format) | 快速 |
| 测试 | pytest | 主流 |
| CI/CD | GitHub Actions | 自动构建发布到 PyPI |

### 2.6 包结构

```
src/
├── handler/           # 接入层
│   ├── cli.py         # CLI 入口 (argparse)
│   ├── mcp.py         # MCP 服务器入口 (FastMCP)
│   └── __init__.py
├── service/           # 业务层
│   ├── __init__.py    # 服务注册表 + 工厂函数
│   ├── base.py        # SubtitleService 抽象基类
│   ├── bilibili.py    # B站服务实现
│   ├── youtube.py     # YouTube 服务实现
│   └── local.py       # 本地文件服务实现
└── core/              # 基础层
    ├── __init__.py
    ├── asr.py         # Whisper ASR 转录
    ├── audio.py       # 音频提取 (ffmpeg)
    ├── browser.py     # 浏览器 Cookie 读取
    ├── cookie.py      # Cookie 管理（统一入口）
    ├── formatter.py   # 字幕格式化 (text/srt/json)
    ├── text.py        # 文本处理（繁简转换、文件名清理）
    └── logging.py     # 日志系统
```

**入口点映射**：
- `video-captions` → `handler.cli:main`
- `video-captions-mcp` → `handler.mcp:main`

---

## 3. 用户场景与预期行为

### 3.1 CLI 场景

| # | 场景 | 命令 | 预期行为 |
|---|------|------|----------|
| S1 | B站有字幕视频 | `video-captions <BV号URL>` | 自动读取浏览器 Cookie，调用 B站 API 获取中文字幕，输出纯文本 |
| S2 | B站无字幕视频 | `video-captions <BV号URL>` | API 无字幕 → 下载视频 → 提取音频 → Whisper ASR → 输出字幕 |
| S3 | YouTube 有字幕视频 | `video-captions <YouTube URL>` | yt-dlp 获取视频信息 → 下载 json3 字幕文件 → 解析输出 |
| S4 | YouTube 无字幕视频 | `video-captions <YouTube URL>` | 字幕列表为空 → 下载视频 → 提取音频 → Whisper ASR → 输出字幕 |
| S5 | 本地视频文件 | `video-captions /path/to/video.mp4` | 直接 ffmpeg 提取音频 → Whisper ASR → 输出字幕 |
| S6 | 本地音频文件 | `video-captions /path/to/audio.mp3` | 跳过音频提取，直接 Whisper ASR → 输出字幕 |
| S7 | 指定输出格式 | `video-captions --format srt <URL>` | 输出 SRT 字幕格式，带时间戳 |
| S8 | 指定 ASR 模型 | `video-captions --model small <URL>` | 使用 small 模型（更快但精度较低） |
| S9 | 指定浏览器 | `video-captions --browser arc <URL>` | 仅从 Arc 读取 Cookie |

### 3.2 MCP 场景

| # | 场景 | 工具 | 预期行为 |
|---|------|------|----------|
| M1 | AI 获取视频字幕 | `download_captions(url=...)` | 返回结构化 JSON，包含 source/format/subtitle_count/content/video_title |
| M2 | AI 转录本地文件 | `transcribe_local_file(file_path=...)` | 返回 ASR 生成的字幕，show_progress=False 避免干扰输出 |
| M3 | 不支持的 URL | `download_captions(url=...)` | 返回 `{"error": "...", "message": "...", "suggestion": "..."}` |

### 3.3 边界场景

| # | 场景 | 预期行为 |
|---|------|----------|
| E1 | 未安装 yt-dlp | ASR 兜底时报错，suggestion 提示安装 |
| E2 | 未安装 ffmpeg | 音频提取失败，提示 `brew install ffmpeg` |
| E3 | 未登录 B站 | Cookie 获取失败 → 环境变量检查 → 抛出异常提示登录 |
| E4 | YouTube 需要登录 | 检测到 `Sign in` 关键字 → 抛出异常 |
| E5 | 文本过长 | text 格式超过 50000 字符时自动截断 |
| E6 | 繁体字幕 | 自动通过 OpenCC 转为简体 |
| E7 | 不支持的来源 | 工厂返回 None，提示支持的平台列表 |

---

## 4. 各服务实现细节

### 4.1 BilibiliService

**Cookie 获取**：
1. 从浏览器 Cookie 读取（通过 `browser-cookie3` 解密 Chromium Cookie）
2. 回退到环境变量 `BILIBILI_SESSDATA`
3. 都没有则抛出异常

**API 字幕获取流程**：
```
get_info(bvid) → 获取 cid
    → x/player/wbi/v2 API → 获取字幕列表
    → 优先选择中文（ai-zh > zh-Hans > zh-CN > zh）
    → 下载字幕 JSON → 解析 body 字段
```

**URL 匹配规则**：
- `bilibili.com/video/` — 标准视频页
- `bilibili.com/list/` — 合集/列表页
- `^BV[\w]+$` — 纯 BV 号

### 4.2 YouTubeService

**字幕获取流程**：
```
yt-dlp --dump-json → 获取字幕语言列表
    → 按优先级选择语言 (zh-Hans-en > zh-Hant-en > zh-Hans > ... > zh > en)
    → yt-dlp --write-subs --sub-format json3 → 下载字幕文件
    → 解析 json3 格式（events 数组 + segs 文本拼接）
```

**URL 匹配规则**：
- `youtube.com/watch` — 标准视频页
- `youtube.com/shorts/` — Shorts 短视频
- `youtu.be/` — 短链接
- `youtube.com/embed/` — 嵌入链接
- `youtube.com/v/` — 旧版链接

**Cookie 传递**：通过 `yt-dlp --cookies-from-browser <browser>` 参数传递，利用 yt-dlp 自身的 Cookie 处理能力。

### 4.3 LocalService

**特点**：
- 不需要 browser 参数（无网络请求）
- 不需要 Cookie
- 仅支持 ASR 模式（无 API 字幕）
- 视频文件先提取音频再 ASR，音频文件直接 ASR

**文件格式支持**：
| 类型 | 格式 |
|------|------|
| 视频 | mp4, avi, mkv, mov, flv, wmv, webm, m4v, mpg, mpeg |
| 音频 | mp3, wav, m4a, aac, flac, ogg, wma, opus |

---

## 5. Core 层模块说明

### 5.1 ASR (asr.py)

使用 `mlx-whisper`（Apple Silicon 优化版 Whisper）进行语音识别。

| 参数 | 说明 |
|------|------|
| 语言 | 固定为 `zh`（中文） |
| 幻觉抑制 | `hallucination_silence_threshold=0.5` |
| 上下文条件 | `condition_on_previous_text=False`（避免错误累积） |

| 模型 | HuggingFace 路径 | 特点 |
|------|-----------------|------|
| base | mlx-community/whisper-base-mlx | 最快，精度较低 |
| small | mlx-community/whisper-small-mlx | 较快 |
| medium | mlx-community/whisper-medium-mlx | 平衡 |
| large | mlx-community/whisper-large-v3-mlx | 精度最高（默认） |

### 5.2 音频提取 (audio.py)

使用 ffmpeg 将视频转为 16kHz 单声道 WAV：

```bash
ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar 16000 -ac 1 <output.wav>
```

参数选择原因：
- `pcm_s16le`：16位 PCM，Whisper 兼容性最好
- `16000 Hz`：Whisper 训练采样率
- `-ac 1`：单声道，减少数据量

### 5.3 Cookie 读取 (browser.py + cookie.py)

**读取策略**（`cookie.py` 统一入口）：
1. 浏览器读取（默认启用）
2. 环境变量 `BILIBILI_SESSDATA` 回退

**浏览器支持**（`browser.py`）：
- Arc 通过 `yt-dlp` 的 Chromium Cookie 解密能力读取，Chrome/Edge/Firefox/Brave 通过 `browser-cookie3` 读取
- 默认浏览器为 Arc；显式传 `auto` 时尝试顺序：Arc → Chrome → Edge → Brave → Firefox
- 记录最后成功的浏览器名称，便于日志输出

### 5.4 字幕格式化 (formatter.py)

| 格式 | 说明 | 截断 |
|------|------|------|
| text | 纯文本，按行拼接 | 超过 50000 字符截断 |
| srt | SRT 字幕格式，带序号和时间戳 | 不截断 |
| json | 结构化 JSON，含 from/to/content | 不截断 |

所有格式输出前统一执行繁简转换（`t2s`）。

### 5.5 日志系统 (logging.py)

统一前缀 `[video-captions]`，输出到 stderr（不污染 stdout 的字幕内容）。

| 级别 | 格式 | 用途 |
|------|------|------|
| info | `[video-captions] message` | 一般信息 |
| step | `[video-captions] ▶ step: message` | 流程步骤 |
| success | `[video-captions] ✓ message` | 操作成功 |
| warning | `[video-captions] ⚠ message` | 警告 |
| error | `[video-captions] ✗ message` | 错误 |
| debug | `[video-captions] └─ message` | 详细调试（仅 -v 模式） |

---

## 6. 错误处理

### 6.1 统一错误响应格式

所有错误通过结构化字典返回，便于 CLI 和 MCP 统一处理：

```python
{
    "error": "错误类型",
    "message": "详细错误信息",
    "suggestion": "解决建议",   # 可选
    "stderr": "原始错误输出"    # 可选，用于 subprocess 错误
}
```

### 6.2 ASR 兜底策略

当平台 API 字幕获取失败时，自动切换到 ASR 模式：

```
API 字幕获取 → 失败/无字幕
    → 下载视频 (yt-dlp)
    → 提取音频 (ffmpeg → 16kHz WAV)
    → Whisper ASR 转录
    → 格式化输出
```

**临时文件管理**：ASR 流程使用 `tempfile.TemporaryDirectory()`，处理完成后自动清理。

---

## 7. 当前已知限制

### 7.1 设计约束（符合预期，不做调整）

| # | 约束 | 说明 |
|---|------|------|
| 1 | ASR 仅限 Apple Silicon | mlx-whisper 依赖 Apple Silicon GPU，Intel Mac 和 Linux 不支持 |
| 2 | 字幕语言固定中文 | Whisper 固定 `language="zh"`，不支持其他语言 |
| 3 | 系统依赖需手动安装 | yt-dlp 和 ffmpeg 需用户自行 `brew install` |
| 4 | Cookie 读取仅 macOS | browser.py 的浏览器路径硬编码为 macOS 路径 |

### 7.2 待优化

| # | 限制 | 影响 | 备注 |
|---|------|------|------|
| 5 | 无字幕缓存 | 同一视频重复调用会重新下载和 ASR | 可考虑本地缓存机制 |
| 6 | 无并发支持 | 多个视频需顺序处理 | ASR 本身是 CPU/GPU 密集型 |
| 7 | 文本格式截断 | 50000 字符硬截断可能丢失内容 | 可改为流式输出或分段返回 |
| 8 | 无进度回调 | CLI 模式下 ASR 进度仅依赖 mlx-whisper 自身输出 | MCP 模式已通过 show_progress=False 规避 |

---

## 8. 演进规划

### 8.1 Phase 1：缓存与性能优化

**目标**：减少重复下载和 ASR 的开销。

- 字幕缓存：相同 URL + 格式 的结果缓存到本地（`~/.cache/video-captions/`）
- 音频缓存：已提取的音频文件保留，避免重复提取
- 缓存过期：基于时间或手动清理

### 8.2 Phase 2：多语言支持

**目标**：支持非中文视频的字幕获取。

- Whisper 语言自动检测（移除 `language="zh"` 固定值）
- 输出格式支持双语字幕
- 字幕翻译集成（可选）

### 8.3 Phase 3：跨平台支持

**目标**：支持 Linux 和 Windows。

- browser.py 适配 Linux/Windows 浏览器 Cookie 路径
- ASR 引擎可选（mlx-whisper / faster-whisper / openai-whisper）
- 系统依赖检测和安装提示

### 8.4 Phase 4：更多平台

**目标**：扩展更多视频平台支持。

- Twitter/X 视频字幕
- 抖音/TikTok 视频字幕
- Vimeo 视频字幕
- 通过服务注册表扩展，无需修改核心逻辑

---

## 9. 技术决策记录

### 9.1 为什么选择 mlx-whisper 而非 openai-whisper？

| 维度 | mlx-whisper | openai-whisper |
|------|-------------|----------------|
| 硬件 | Apple Silicon (M1/M2/M3/M4) | 通用 GPU |
| 速度 | 快（Metal 加速） | 慢（CPU 或需 CUDA） |
| 安装 | pip install | pip install + torch |
| 依赖 | 轻量 | 重（PyTorch ~2GB） |
| 适用 | macOS 个人工具 | 服务器/跨平台 |

结论：作为本地工具，优先考虑 Apple Silicon 性能和安装简便性。

### 9.2 为什么用 subprocess 调用 yt-dlp/ffmpeg 而非 Python 库？

| 维度 | subprocess 调用 | Python 库 |
|------|----------------|-----------|
| 兼容性 | 使用系统安装的版本，自动更新 | 需要额外 Python 依赖 |
| 功能完整性 | 完整 CLI 功能 | 库可能缺少部分功能 |
| 格式支持 | ffmpeg 支持所有格式 | 依赖库的解码器 |
| 错误处理 | stderr 输出详细 | 异常信息可能不完整 |

结论：yt-dlp 和 ffmpeg 本身就是 CLI 工具，subprocess 调用最简单直接。

### 9.3 为什么日志输出到 stderr？

- stdout 用于字幕内容输出（CLI 模式）或结构化 JSON（MCP 模式）
- stderr 用于进度和状态信息，不污染输出内容
- 用户可以通过管道 `video-captions <URL> > output.txt` 只获取字幕内容

### 9.4 为什么 B站和 YouTube 的 Cookie 策略不同？

| 维度 | B站 | YouTube |
|------|-----|---------|
| Cookie 用途 | SESSDATA（API 认证） | 视频下载认证 |
| 读取方式 | Arc 走 yt-dlp Cookie 解密，其他浏览器走 browser-cookie3 → Python httpx 发送 | Arc 导出临时 cookie 文件，其他指定浏览器走 yt-dlp `--cookies-from-browser` |
| 原因 | B站 API 需要手动构建 HTTP 请求 | yt-dlp 内置了完整的 YouTube Cookie 处理 |

结论：B站因为直接调用 API，需要自己获取和解密 Cookie；YouTube 的 Cookie 处理完全委托给 yt-dlp。

---

## 10. 版本管理与发布

### 10.1 版本号规则

格式：`0.1.YYYYMMDD.序号`

- 推送到 main 分支自动触发 GitHub Actions
- 同一天内多次推送，序号自动递增（.1, .2, .3...）
- 新的一天从 .1 重新开始

### 10.2 发布流程

```
git push main
    → GitHub Actions: 自动版本号递增
    → GitHub Actions: uv build
    → GitHub Actions: pypi-publish (Trusted Publishing)
    → 用户: pip install video-captions / uv tool install video-captions
```

### 10.3 安装方式

```bash
# PyPI 安装
pip install video-captions
uv tool install video-captions

# 本地开发
git clone https://github.com/LuShan123888/Video-Captions.git
cd Video-Captions
uv sync
```
