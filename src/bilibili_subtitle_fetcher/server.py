#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站字幕 MCP 服务器

提供获取B站视频信息和字幕的工具。
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict

from mcp.server.fastmcp import FastMCP
from .api import (
    get_video_info,
    list_subtitles,
    download_subtitle_content,
    ResponseFormat,
    CHARACTER_LIMIT,
    make_safe_filename,
    get_sessdata,
    convert_to_simplified
)

# 初始化MCP服务器
mcp = FastMCP("bilibili_mcp")


# ============================================================================
# Pydantic 输入模型
# ============================================================================

class VideoUrlInput(BaseModel):
    """视频URL输入模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    url: str = Field(
        ...,
        description="B站视频URL，例如: https://www.bilibili.com/video/BV1xx411c7mD/ 或 BV1xx411c7mD",
        min_length=1,
        max_length=500
    )

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """验证URL格式"""
        v = v.strip()
        if not v:
            raise ValueError("URL不能为空")
        # 确保包含BV号
        if 'BV' not in v and 'bv' not in v:
            raise ValueError("URL必须包含有效的B站视频BV号")
        return v


class SubtitleFormat(str, Enum):
    """字幕格式"""
    TEXT = "text"
    SRT = "srt"
    JSON = "json"


class DownloadSubtitlesInput(BaseModel):
    """下载字幕输入模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    url: str = Field(
        ...,
        description="B站视频URL，例如: https://www.bilibili.com/video/BV1xx411c7mD/",
        min_length=1,
        max_length=500
    )
    format: SubtitleFormat = Field(
        default=SubtitleFormat.TEXT,
        description="输出格式: text=纯文本, srt=SRT字幕格式, json=JSON结构化数据"
    )
    sessdata: Optional[str] = Field(
        default=None,
        description="可选的B站SESSDATA认证token（优先使用环境变量 BILIBILI_SESSDATA）",
        max_length=500
    )

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL不能为空")
        if 'BV' not in v and 'bv' not in v:
            raise ValueError("URL必须包含有效的B站视频BV号")
        return v


class ListSubtitlesInput(BaseModel):
    """列出字幕输入模型"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    url: str = Field(
        ...,
        description="B站视频URL",
        min_length=1,
        max_length=500
    )
    sessdata: Optional[str] = Field(
        default=None,
        description="可选的B站SESSDATA认证token（优先使用环境变量 BILIBILI_SESSDATA）",
        max_length=500
    )

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL不能为空")
        if 'BV' not in v and 'bv' not in v:
            raise ValueError("URL必须包含有效的B站视频BV号")
        return v


# ============================================================================
# 工具定义
# ============================================================================

@mcp.tool(
    name="bilibili_get_video_info",
    annotations={
        "title": "获取B站视频信息",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def bilibili_get_video_info(params: VideoUrlInput) -> dict:
    '''获取B站视频的基本信息，包括标题、时长、UP主等。

    此工具用于查询B站视频的元数据信息，不下载任何字幕内容。
    如果需要下载字幕，请使用 bilibili_download_subtitles 工具。

    Args:
        params (VideoUrlInput): 包含以下字段的输入参数:
            - url (str): B站视频URL，例如: https://www.bilibili.com/video/BV1xx411c7mD/

    Returns:
        str: JSON格式的视频信息，包含以下字段:

        成功响应:
        {
            "title": str,        # 视频标题
            "bvid": str,         # BV号
            "cid": int,          # 分P ID
            "duration": int,     # 视频时长（秒）
            "description": str,  # 视频简介
            "owner": {
                "name": str,     # UP主名称
                "mid": int       # UP主ID
            },
            "has_subtitle": bool # 是否有字幕
        }

        错误响应:
        "Error: <错误信息>"

    Examples:
        - 使用场景: "获取这个B站视频的信息" -> params with url="https://www.bilibili.com/video/BV1xx411c7mD/"
        - 使用场景: "这个视频有多长？" -> params with url="BV1xx411c7mD"
        - 不要使用: 需要下载字幕时（使用 bilibili_download_subtitles 代替）

    Error Handling:
        - 返回 "Error: 无法获取视频信息" 如果视频不存在或URL无效
        - 返回 "Error: B站API返回错误" 如果B站API调用失败
    '''
    try:
        info = await get_video_info(params.url)
        # 直接返回字典，让FastMCP处理序列化
        return info

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"获取视频信息时发生错误: {type(e).__name__}", "message": str(e)}


@mcp.tool(
    name="bilibili_list_subtitles",
    annotations={
        "title": "列出可用字幕",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def bilibili_list_subtitles(params: ListSubtitlesInput) -> dict:
    '''列出视频可用的字幕语言和类型。

    此工具用于查询视频有哪些字幕可选，不下载字幕内容。
    对于需要登录的AI字幕，需要设置SESSDATA（环境变量或参数）。

    Args:
        params (ListSubtitlesInput): 包含以下字段的输入参数:
            - url (str): B站视频URL
            - sessdata (Optional[str]): 可选的SESSDATA认证token

    Returns:
        str: JSON格式的字幕列表信息:

        成功响应:
        {
            "available": bool,      # 是否有可用字幕
            "subtitles": [
                {
                    "lan": str,          # 语言代码 (如 "ai-zh")
                    "lan_doc": str,      # 语言描述 (如 "中文（AI生成）")
                    "subtitle_url": str  # 字幕下载地址
                }
            ],
            "subtitle_count": int   # 字幕数量
        }

        无字幕响应:
        {
            "available": false,
            "subtitles": [],
            "subtitle_count": 0
        }

    Examples:
        - 使用场景: "这个视频有哪些字幕？"
        - 使用场景: "检查视频是否有AI字幕"
        - 不要使用: 需要字幕内容时（使用 bilibili_download_subtitles 代替）

    Error Handling:
        - 返回包含 "error" 字段的JSON如果请求失败
        - 如果需要登录才能获取字幕，提示设置 SESSDATA
    '''
    try:
        result = await list_subtitles(params.url, params.sessdata)
        # 直接返回字典，让FastMCP处理序列化
        return result

    except Exception as e:
        return {
            "available": False,
            "subtitles": [],
            "subtitle_count": 0,
            "error": str(e)
        }


@mcp.tool(
    name="bilibili_download_subtitles",
    annotations={
        "title": "下载视频字幕",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def bilibili_download_subtitles(params: DownloadSubtitlesInput) -> dict:
    '''下载B站视频字幕内容，支持多种格式。

    此工具会下载完整的字幕内容。对于长视频，内容可能会被截断。
    优先使用B站API字幕，对于无字幕视频目前不支持ASR（需要额外工具）。

    Args:
        params (DownloadSubtitlesInput): 包含以下字段的输入参数:
            - url (str): B站视频URL
            - format (SubtitleFormat): 输出格式
                - "text": 纯文本，适合阅读和总结
                - "srt": SRT字幕格式，适合视频播放
                - "json": 结构化JSON数据，适合程序处理
            - sessdata (Optional[str]): 可选的SESSDATA认证token

    Returns:
        str: 根据format参数返回不同格式的内容:

        text格式 - 纯文本内容:
        {
            "source": str,           # "bilibili_api"
            "format": str,           # "text"
            "subtitle_count": int,   # 字幕条数
            "content": str,          # 字幕文本内容
            "video_title": str       # 视频标题
        }

        srt格式 - SRT字幕:
        {
            "source": str,
            "format": "srt",
            "subtitle_count": int,
            "content": str,          # SRT格式字幕
            "video_title": str
        }

        json格式 - 结构化数据:
        {
            "source": str,
            "format": "json",
            "subtitle_count": int,
            "subtitles": [         # 字幕数组
                {
                    "from": float,  # 开始时间（秒）
                    "to": float,    # 结束时间（秒）
                    "content": str  # 字幕内容
                }
            ],
            "video_title": str
        }

        无字幕错误:
        {
            "error": str,
            "suggestion": str
        }

    Examples:
        - 使用场景: "下载这个视频的字幕" -> url="...", format="text"
        - 使用场景: "获取SRT格式字幕" -> url="...", format="srt"
        - 使用场景: "总结这个视频的内容" -> url="...", format="text"
        - 不要使用: 只需要视频信息时（使用 bilibili_get_video_info）

    Error Handling:
        - 返回包含 "error" 字段的JSON如果视频无字幕
        - 内容超过字符限制时会截断并添加提示
        - 需要登录的字幕会提示设置 SESSDATA
    '''
    try:
        result = await download_subtitle_content(
            params.url,
            ResponseFormat(params.format.value),
            params.sessdata
        )

        # 直接返回字典，让FastMCP处理序列化
        return result

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
