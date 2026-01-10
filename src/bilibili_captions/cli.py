# /// script
# dependencies = ["httpx", "faster-whisper", "opencc-python-reimplemented"]
# -*-

"""
B站字幕抓取工具 - CLI 版本

支持从B站视频下载字幕，若无字幕则使用 Whisper ASR 生成。
"""

import asyncio
import sys

from .core import (
    download_subtitles_with_asr,
    get_video_info,
    require_sessdata,
    ResponseFormat,
    get_sessdata,
)


def print_result(result: dict) -> None:
    """格式化打印字幕结果"""
    if "error" in result:
        print(f"\n错误: {result.get('error', '未知错误')}")
        if "message" in result:
            print(f"详情: {result['message']}")
        if "suggestion" in result:
            print(f"提示: {result['suggestion']}")
        return None

    video_title = result.get("video_title", "未知")
    source = result.get("source", "unknown")
    source_label = "B站AI字幕 (API直接获取)" if source == "bilibili_api" else "Whisper ASR语音识别 (AI生成)"
    subtitle_count = result.get("subtitle_count", 0)

    print(f"\n{'='*60}")
    print(f"字幕来源: {source_label}")
    print(f"视频标题: {video_title}")
    print(f"{'='*60}")

    if "content" in result:
        print(result["content"])
    elif "subtitles" in result:
        # JSON 格式
        for item in result["subtitles"]:
            print(item.get("content", ""))

    print(f"{'='*60}")
    print(f"\n共 {subtitle_count} 条字幕")
    return None


def main() -> None:
    """CLI入口点"""
    if len(sys.argv) < 2:
        print("用法: bilibili-captions <B站视频URL> [模型大小]")
        print("模型大小可选: base, small, medium (默认), large")
        sys.exit(1)

    video_url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"

    # 验证模型大小
    valid_models = ["base", "small", "medium", "large"]
    if model_size not in valid_models:
        print(f"警告: 无效的模型大小 '{model_size}'，使用默认 'medium' 模型")
        model_size = "medium"

    print(f"使用模型: {model_size}")

    # 检查 SESSDATA
    try:
        require_sessdata()
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 获取视频信息
    try:
        info = asyncio.run(get_video_info(video_url))
        print(f"视频标题: {info.get('title')}")
    except Exception as e:
        print(f"错误: 无法获取视频信息 - {e}")
        sys.exit(1)

    # 下载字幕（API优先，ASR兜底）
    sessdata = get_sessdata()
    result = asyncio.run(download_subtitles_with_asr(
        video_url,
        ResponseFormat.TEXT,
        model_size,
        sessdata
    ))

    print_result(result)


if __name__ == "__main__":
    main()
