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
工作方式:
  在线视频  优先获取平台字幕；没有字幕时自动下载音频并使用 Whisper ASR 转写
  本地文件  直接使用 Whisper ASR 转写；--model 仅在 ASR 转写时生效
  Cookie    --browser 仅用于 B站或 YouTube 登录态，默认读取 Arc

输出约定:
  text/srt  字幕内容写入 stdout
  json      单个 JSON 对象写入 stdout，包含 source、format、subtitle_count、
            subtitles、video_title，以及可选的 language
  日志/错误  写入 stderr；成功退出码为 0，失败退出码非 0
  Agent/脚本建议使用 --format json；--verbose 不会改变 stdout 的数据格式

依赖:
  在线视频下载需要 yt-dlp；视频音轨提取需要 ffmpeg；Whisper ASR 仅支持 Apple Silicon

示例:
  video-captions "https://www.bilibili.com/video/BV1xx"
  video-captions --format srt "https://youtu.be/xxx"
  video-captions --format json "https://youtube.com/watch?v=xxx"
  video-captions --browser chrome "https://www.bilibili.com/video/BV1xx"
  video-captions --model small "/path/to/audio.mp3\"""",
    )
    parser.add_argument("source", help="B站/YouTube URL、B站 BV 号或本地音视频文件路径")
    parser.add_argument(
        "--browser",
        choices=["auto", "arc", "chrome", "edge", "firefox", "brave"],
        default="arc",
        help="在线视频需要登录时读取 Cookie；对本地文件无效（默认 arc）",
    )
    parser.add_argument(
        "--model",
        choices=["base", "small", "medium", "large"],
        default="large",
        help="ASR 回退或本地转写使用的 Whisper 模型（默认 large）",
    )
    parser.add_argument(
        "--format",
        choices=["text", "srt", "json"],
        default="text",
        help="stdout 输出格式；Agent/脚本建议使用 json（默认 text）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="将详细日志和元信息写入 stderr"
    )

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
