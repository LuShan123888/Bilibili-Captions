# /// script
# dependencies = ["httpx", "mlx-whisper", "opencc-python-reimplemented"]
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

    # 打印字幕内容（ASR 已通过 verbose 输出，无需重复打印）
    source = result.get("source", "")
    if source != "whisper_asr":
        content = result.get("content")
        if content:
            print(content)

    subtitle_count = result.get("subtitle_count", 0)
    print(f"\n共 {subtitle_count} 条字幕")
    return None


def main() -> None:
    """CLI入口点"""
    if len(sys.argv) < 2:
        print("用法: bilibili-captions <B站视频URL> [模型大小]")
        print("模型大小可选: base, small, medium, large, large-v3 (默认)")
        sys.exit(1)

    video_url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "large-v3"

    # 验证模型大小
    valid_models = ["base", "small", "medium", "large", "large-v3"]
    if model_size not in valid_models:
        print(f"警告: 无效的模型大小 '{model_size}'，使用默认 'large-v3' 模型")
        model_size = "large-v3"

    # 检查 SESSDATA
    try:
        require_sessdata()
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 获取视频信息并显示头部
    try:
        info = asyncio.run(get_video_info(video_url))
        video_title = info.get('title', '未知')

        # 显示字幕来源（先用 API 尝试，失败则用 ASR）
        has_subtitle = info.get('has_subtitle', False)
        source_label = "B站AI字幕 (API直接获取)" if has_subtitle else "Whisper ASR语音识别 (AI生成，如无API字幕)"

        print(f"{'='*60}")
        print(f"视频标题: {video_title}")
        print(f"字幕来源: {source_label}")
        print(f"{'='*60}\n")
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
