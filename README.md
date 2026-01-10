# bilibili-captions

B站字幕下载工具，支持 API 获取和 Whisper ASR 自动生成。

## 功能特性

- 🎬 **API 下载** - 直接从 B站 API 获取视频字幕
- 🤖 **ASR 生成** - 无字幕时自动使用 Whisper 生成
- 🌏 **繁简转换** - 自动转换为简体中文
- 📦 **MCP 服务器** - 集成到 Claude Desktop
- 🧪 **完整测试** - 包含真实视频测试用例

---

## 使用

### 安装

```bash
# 使用 pip 安装（推荐）
pip install bilibili-captions

# 或使用 uv
uv pip install bilibili-captions
```

### 运行

```bash
# 1. 设置 SESSDATA 环境变量
export BILIBILI_SESSDATA="你的值"

# 2. 运行命令
bilibili-captions <BV号或URL> [模型大小]

# 示例
bilibili-captions BV16YC3BrEDz
bilibili-captions https://www.bilibili.com/video/BV1qViQBwELr medium
```

**模型大小选项：**
- `base` - 最快，精度较低
- `small` - 较快
- `medium` - 平衡（默认）
- `large` - 最慢，精度最高

### MCP 服务器

在 Claude Desktop 的 `claude_desktop_config.json` 中配置：

```json
{
  "mcpServers": {
    "bilibili-captions": {
      "command": "uvx",
      "args": ["bilibili-captions-mcp"],
      "env": {
        "BILIBILI_SESSDATA": "你的 SESSDATA"
      },
      "timeout": 600000
    }
  }
}
```

### MCP 工具

#### download_captions

下载 B 站视频字幕，支持多种格式。

| 参数 | 类型 | 说明 |
|------|------|------|
| `url` | 必需 | B站视频URL或BV号 |
| `format` | 可选 | `text`(默认) / `srt` / `json` |
| `model_size` | 可选 | `base` / `small` / `medium`(默认) / `large` |

**返回示例：**
```json
{
  "source": "bilibili_api",
  "format": "text",
  "subtitle_count": 189,
  "content": "字幕内容...",
  "video_title": "视频标题"
}
```

---

## 开发

### 项目结构

```
bilibili_subtitle_fetcher/
├── src/bilibili_captions/
│   ├── __init__.py
│   ├── core.py      # 核心 API 功能
│   ├── cli.py       # CLI 入口
│   └── server.py    # MCP 服务器
├── tests/
│   └── test_videos.py    # 测试用例
├── .env.example          # 配置示例
├── pyproject.toml
└── README.md
```

### 本地运行

```bash
# 克隆项目
git clone <repo_url>
cd bilibili_subtitle_fetcher

# 安装依赖
uv sync

# 设置 SESSDATA
cp .env.example .env
# 编辑 .env 填入 SESSDATA

# 运行 CLI
uv run bilibili-captions <URL>

# 运行 MCP 服务器
uv run bilibili-captions-mcp
```

### 测试

项目包含两个真实视频的测试用例：

| 视频 | 场景 | 预期结果 |
|------|------|---------|
| BV16YC3BrEDz | 有 API 字幕 | 189 条，来源 `bilibili_api` |
| BV1qViQBwELr | 无字幕 ASR | 30 条，来源 `whisper_asr` |

```bash
uv run python tests/test_videos.py
# 或
pytest tests/test_videos.py
```

---

## 配置

### SESSDATA 获取

1. 登录 [B站](https://www.bilibili.com/)
2. F12 → Application → Cookies → `SESSDATA`
3. 复制值到环境变量或 `.env` 文件

```bash
# 方式1: 环境变量
export BILIBILI_SESSDATA="你的值"

# 方式2: .env 文件
cp .env.example .env
# 编辑 .env 填入 SESSDATA
```

## 依赖

| 依赖 | 用途 |
|------|------|
| `httpx` | HTTP 客户端 |
| `faster-whisper` | 语音识别（推荐） |
| `openai-whisper` | 语音识别备选 |
| `opencc-python-reimplemented` | 繁简转换 |
| `yt-dlp` | 视频下载（系统） |
| `ffmpeg` | 音频提取（系统） |

### 系统依赖

```bash
# macOS
brew install yt-dlp ffmpeg

# Linux
apt install yt-dlp ffmpeg
```

## 许可证

MIT
