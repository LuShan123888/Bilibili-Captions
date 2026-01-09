# B站字幕 MCP 服务器

B站字幕抓取工具及MCP服务器，支持从B站视频下载AI字幕或使用Whisper ASR生成字幕。

## 功能特性

- 🎬 **下载B站AI字幕** - 直接从B站API获取视频字幕
- 🤖 **Whisper ASR生成** - 对无字幕视频自动生成字幕
- 🌏 **繁简转换** - 自动将繁体字幕转换为简体中文
- 📡 **MCP服务器** - 可作为MCP服务器集成到Claude Desktop等工具

## 安装

### 使用 uv (推荐)

```bash
# 克隆项目
git clone https://github.com/your-username/bilibili-subtitle-fetcher.git
cd bilibili-subtitle-fetcher

# 安装依赖
uv sync
```

### 使用 pip

```bash
pip install -e .
```

## 配置

### SESSDATA 设置

获取B站AI字幕需要登录认证，请按以下步骤获取 `SESSDATA`：

1. 登录 [B站官网](https://www.bilibili.com/)
2. 按 F12 打开开发者工具
3. 进入 Application/Storage -> Cookies
4. 找到 `SESSDATA` 的值

### 环境变量

**方式1: 环境变量**
```bash
export BILIBILI_SESSDATA="your SESSDATA value"
```

**方式2: .env 文件**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 SESSDATA
```

## 使用方式

### CLI 命令行

```bash
# 下载字幕
uv run python main.py https://www.bilibili.com/video/BV1xx411c7mD/
```

### MCP 服务器

#### 方式1: stdio (适用于Claude Desktop)

在 Claude Desktop 配置文件 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "bilibili": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/bilibili_subtitle_fetcher",
        "python",
        "run_mcp.py"
      ],
      "env": {
        "BILIBILI_SESSDATA": "your SESSDATA value"
      }
    }
  }
}
```

#### 方式2: 使用 Python 直接运行

```json
{
  "mcpServers": {
    "bilibili": {
      "command": "python3",
      "args": ["/path/to/bilibili_subtitle_fetcher/run_mcp.py"],
      "env": {
        "BILIBILI_SESSDATA": "your SESSDATA value"
      }
    }
  }
}
```

## MCP 工具列表

### 1. bilibili_get_video_info

获取B站视频的基本信息。

```json
{
  "url": "https://www.bilibili.com/video/BV1xx411c7mD/"
}
```

**返回示例：**
```json
{
  "title": "视频标题",
  "bvid": "BV1xx411c7mD",
  "cid": 123456,
  "duration": 180,
  "description": "视频简介",
  "owner": {"name": "UP主名称", "mid": 12345}
}
```

### 2. bilibili_list_subtitles

列出视频可用的字幕。

```json
{
  "url": "https://www.bilibili.com/video/BV1xx411c7mD/"
}
```

### 3. bilibili_download_subtitles

下载视频字幕内容。

```json
{
  "url": "https://www.bilibili.com/video/BV1xx411c7mD/",
  "format": "text"
}
```

**参数说明：**
- `url`: B站视频URL
- `format`: 输出格式
  - `text`: 纯文本（默认）
  - `srt`: SRT字幕格式
  - `json`: JSON结构化数据

## 开发

```bash
# 安装开发依赖
uv sync

# 运行测试
uv run pytest

# 代码格式化
uv run ruff check
```

## 依赖项

- **mcp** - MCP 协议支持
- **httpx** - 异步HTTP客户端
- **faster-whisper** - 快速语音识别
- **opencc-python-reimplemented** - 繁简中文转换

## 常见问题

### Q: SESSDATA 在哪里获取？

A: 登录B站后，在浏览器开发者工具的 Application -> Cookies 中找到 `SESSDATA`。

### Q: 提示 "未找到SESSDATA认证信息" 怎么办？

A: 请确保已通过以下方式之一设置：
1. 设置环境变量 `BILIBILI_SESSDATA`
2. 在项目目录创建 `.env` 文件
3. MCP配置中设置环境变量

### Q: 支持哪些视频链接格式？

A: 支持标准B站视频链接：
- `https://www.bilibili.com/video/BV1xx411c7mD/`
- `https://b23.tv/xxxxx` (短链接)
- 任何包含BV号的链接

## 许可证

MIT License

## 相关项目

- [bilibili-video-info-mcp](https://github.com/lesir831/bilibili-video-info-mcp) - 获取B站视频弹幕和评论
