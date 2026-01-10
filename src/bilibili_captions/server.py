#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站字幕 MCP 服务器

提供获取B站视频字幕的工具，优先使用API，无字幕时使用ASR生成。
"""

from typing import Literal
from mcp.server.fastmcp import FastMCP
from .core import (
    download_subtitles_with_asr,
    ResponseFormat,
    get_sessdata,
)

# 初始化MCP服务器
mcp = FastMCP("bilibili-captions")


# ============================================================================
# 工具定义
# ============================================================================


@mcp.tool()
async def download_captions(
    url: str,
    format: Literal["text", "srt", "json"] = "text",
    model_size: Literal["base", "small", "medium", "large"] = "medium"
) -> dict:
    """下载B站视频字幕内容，支持多种格式。

    优先从B站API获取字幕，若无字幕则使用Whisper ASR自动生成。

    Args:
        url: B站视频URL（支持多种格式：完整URL、BV号、稍后观看链接等）
        format: 输出格式
            - "text": 纯文本，适合阅读和总结
            - "srt": SRT字幕格式，适合视频播放
            - "json": 结构化JSON数据，适合程序处理
        model_size: ASR模型大小（当API无字幕时使用）
            - "base": 最快，精度较低
            - "small": 较快
            - "medium": 平衡（默认）
            - "large": 最慢，精度最高

    Returns:
        成功时:
        {
            "source": "bilibili_api" | "whisper_asr",  # 字幕来源
            "format": str,
            "subtitle_count": int,
            "content": str,          # text/srt格式
            "video_title": str
        }

        错误时:
        {
            "error": str,
            "message": str
        }

    Note:
        - 需要在 MCP 配置中设置 BILIBILI_SESSDATA 环境变量以获取 AI 字幕
        - ASR 兜底需要安装 yt-dlp 和 ffmpeg
        - ASR 处理可能需要几分钟，请在 Claude Desktop 中增加超时时间
        - 所有字幕输出自动转换为简体中文
    """
    try:
        sessdata = get_sessdata()
        # MCP 服务器禁用进度条（无终端输出）
        return await download_subtitles_with_asr(
            url, ResponseFormat(format), model_size, sessdata, show_progress=False
        )
    except Exception as e:
        return {
            "error": f"下载字幕时发生错误: {type(e).__name__}",
            "message": str(e)
        }


# ============================================================================
# 主入口
# ============================================================================

def main() -> None:
    """MCP服务器入口点"""
    mcp.run()


if __name__ == "__main__":
    main()
