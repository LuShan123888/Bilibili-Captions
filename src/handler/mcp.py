#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频字幕 MCP Server Handler

处理 MCP 协议，调用 Service 层完成字幕下载
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP

from service import get_service
from core.formatter import ResponseFormat

# 初始化 MCP 服务器
mcp = FastMCP("video-captions")


@mcp.tool()
async def download_captions(
        url: str,
        format: Literal["text", "srt", "json"] = "text",
        model_size: Literal["base", "small", "medium", "large"] = "large",
        browser: Literal["auto", "arc", "chrome", "edge", "firefox", "brave"] = "arc"
) -> dict:
    """下载视频字幕内容，支持多种格式。

    支持平台：B站、YouTube
    优先从平台 API 获取字幕，若无字幕则使用 ASR 生成。

    Args:
        url: 视频 URL
            - B站: https://www.bilibili.com/video/BV1xx... 或 BV号
            - YouTube: https://www.youtube.com/watch?v=... 或 youtu.be/...
        format: 输出格式
            - "text": 纯文本，适合阅读和总结
            - "srt": SRT字幕格式，适合视频播放
            - "json": 结构化JSON数据，适合程序处理
        model_size: ASR 模型大小（当 API 无字幕时使用）
            - "base": 最快，精度较低
            - "small": 较快
            - "medium": 平衡
            - "large": 精度最高（默认，mlx-whisper 优化）
        browser: 从哪个浏览器读取 Cookie
            - "arc": 仅从 Arc 读取（默认）
            - "auto": 自动尝试所有浏览器
            - "chrome": 仅从 Chrome 读取
            - "edge": 仅从 Edge 读取
            - "firefox": 仅从 Firefox 读取
            - "brave": 仅从 Brave 读取

    Returns:
        成功时:
        {
            "source": "bilibili_api" | "youtube_api" | "whisper_asr",
            "format": str,
            "subtitle_count": int,
            "content": str,
            "video_title": str
        }

        错误时:
        {
            "error": str,
            "message": str
        }
    """
    try:
        service = get_service(url, browser)

        if not service:
            return {
                "error": "不支持的 URL 格式",
                "message": f"不支持的 URL: {url}",
                "suggestion": "支持的平台：B站 (bilibili.com)、YouTube (youtube.com)"
            }

        result = await service.download_subtitle(url, ResponseFormat(format), model_size=model_size)
        return result

    except Exception as e:
        return {
            "error": f"下载字幕时发生错误: {type(e).__name__}",
            "message": str(e)
        }


@mcp.tool()
async def transcribe_local_file(
        file_path: str,
        format: Literal["text", "srt", "json"] = "text",
        model_size: Literal["base", "small", "medium", "large"] = "large"
) -> dict:
    """对本地音频/视频文件进行 ASR 语音识别生成字幕。

    使用 Whisper ASR 对本地文件进行语音识别，生成中文字幕。

    Args:
        file_path: 本地文件路径
            - 音频格式: mp3, wav, m4a, aac, flac, ogg, wma, opus
            - 视频格式: mp4, avi, mkv, mov, flv, wmv, webm, m4v
        format: 输出格式
            - "text": 纯文本，适合阅读和总结
            - "srt": SRT字幕格式，适合视频播放
            - "json": 结构化JSON数据，适合程序处理
        model_size: ASR 模型大小
            - "base": 最快，精度较低
            - "small": 较快
            - "medium": 平衡（默认）
            - "large": 精度最高（mlx-whisper 优化）

    Returns:
        成功时:
        {
            "source": "whisper_asr",
            "format": str,
            "subtitle_count": int,
            "content": str,
            "video_title": str
        }

        错误时:
        {
            "error": str,
            "message": str,
            "suggestion": str
        }
    """
    try:
        from service.local import LocalService

        service = LocalService()

        if not service.is_supported(file_path):
            return {
                "error": "不支持的文件格式",
                "message": f"不支持的文件格式: {file_path}",
                "suggestion": "支持的音频格式: mp3, wav, m4a, aac, flac, ogg, wma, opus; "
                              "支持的视频格式: mp4, avi, mkv, mov, flv, wmv, webm, m4v"
            }

        return await service.download_subtitle(
            file_path, ResponseFormat(format), model_size=model_size, show_progress=False
        )
    except Exception as e:
        return {
            "error": f"ASR转录时发生错误: {type(e).__name__}",
            "message": str(e)
        }


def main() -> None:
    """MCP 服务器入口点"""
    mcp.run()


if __name__ == "__main__":
    main()
