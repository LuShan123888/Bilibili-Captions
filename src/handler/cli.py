"""
视频字幕抓取工具 - CLI Handler

处理 CLI 命令行输入，调用 Service 层完成字幕下载
"""

import argparse
import asyncio
import json
import sys

from service import get_service
from core.formatter import ResponseFormat
from core.logging import log_info, set_verbose_log


def print_result(result: dict, format: ResponseFormat, verbose: bool) -> None:
    """格式化打印字幕结果"""
    if "error" in result:
        print(f"错误: {result.get('error', '未知错误')}", file=sys.stderr)
        if "message" in result:
            print(f"详情: {result['message']}", file=sys.stderr)
        if "suggestion" in result:
            print(f"提示: {result['suggestion']}", file=sys.stderr)
        sys.exit(1)

    if format == ResponseFormat.JSON:
        print(json.dumps(result, ensure_ascii=False))
        return

    content = result.get("content")
    if content:
        print(content)

    if verbose:
        video_title = result.get("video_title", "未知")
        subtitle_count = result.get("subtitle_count", 0)
        print(f"\n视频标题: {video_title}", file=sys.stderr)
        print(f"共 {subtitle_count} 条字幕", file=sys.stderr)


def main() -> None:
    """CLI 入口点"""
    parser = argparse.ArgumentParser(
        description="视频字幕下载工具，支持 B站、YouTube 和本地文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  video-captions https://www.bilibili.com/video/BV1xx
  video-captions --format json https://youtube.com/watch?v=xxx
  video-captions --browser arc --format srt /path/to/video.mp4
  video-captions --model small -v https://youtu.be/xxx""",
    )
    parser.add_argument("source", help="视频 URL 或本地文件路径")
    parser.add_argument(
        "--browser",
        choices=["auto", "arc", "chrome", "edge", "firefox", "brave"],
        default="arc",
        help="从浏览器读取 Cookie（默认 arc）",
    )
    parser.add_argument(
        "--model",
        choices=["base", "small", "medium", "large"],
        default="large",
        help="Whisper ASR 模型大小（默认 large）",
    )
    parser.add_argument(
        "--format", choices=["text", "srt", "json"], default="text", help="输出格式（默认 text）"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志和元信息")

    args = parser.parse_args()

    if args.verbose:
        set_verbose_log(True)
        log_info("详细日志模式已启用")

    service = get_service(args.source, args.browser)
    if not service:
        print(f"错误: 不支持的来源: {args.source}", file=sys.stderr)
        print("支持的平台: B站、YouTube、本地音频/视频文件", file=sys.stderr)
        sys.exit(1)

    log_info(f"检测到平台: {service.name}")

    format = ResponseFormat(args.format)

    # 下载字幕
    result = asyncio.run(service.download_subtitle(args.source, format, model_size=args.model))
    print_result(result, format, args.verbose)


if __name__ == "__main__":
    main()
