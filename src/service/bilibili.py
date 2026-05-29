"""
B站服务 - 字幕下载和处理
"""

import logging
import os
import re
import subprocess
import tempfile
import urllib.parse
from typing import Dict, Any, Optional

import httpx

from .base import SubtitleService
from core.formatter import ResponseFormat, format_subtitle
from core.audio import extract_audio
from core.asr import transcribe_with_asr
from core.logging import (
    log_success,
    log_warning,
    log_step,
)
from core.text import make_safe_filename
from core.cookie import get_sessdata

# 禁用 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)


API_BASE_URL = "https://api.bilibili.com"


class BilibiliService(SubtitleService):
    """B站字幕服务"""

    def __init__(self, browser: Optional[str] = "auto"):
        self.browser = browser
        self._sessdata = None

    @property
    def name(self) -> str:
        return "bilibili"

    def is_supported(self, source: str) -> bool:
        """判断是否为 B站视频"""
        patterns = [
            r'bilibili\.com/video/',
            r'bilibili\.com/list/',
            r'^BV[\w]+$',
        ]
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return True
        return False

    def _extract_bvid(self, url: str) -> str:
        """从 URL 中提取 BV 号"""
        source = url.strip()
        if source.startswith('BV'):
            return source.split('?')[0].split('/')[0]

        parsed = urllib.parse.urlparse(source)
        params = urllib.parse.parse_qs(parsed.query)
        query_bvid = None
        for key, values in params.items():
            if key.lower() == 'bvid' and values:
                query_bvid = values[0]
                break

        path = parsed.path or source
        if '/list/watchlater' in path.lower() and query_bvid:
            return query_bvid

        video_match = re.search(r'/video/(BV[a-zA-Z0-9]+)', path, re.IGNORECASE)
        if video_match:
            return video_match.group(1)

        if query_bvid:
            return query_bvid

        last_part = path.rstrip('/').split('/')[-1].split('?')[0]
        if last_part.startswith('BV'):
            return last_part
        raise ValueError(f"无法从 URL 中提取 BV 号: {source}")

    def _ensure_sessdata(self) -> str:
        """确保获取 SESSDATA，如果没有则抛出异常"""
        if self._sessdata:
            return self._sessdata

        sessdata = get_sessdata(self.browser)
        if not sessdata:
            raise ValueError(
                "未找到 B站 SESSDATA。请通过以下方式之一提供：\n"
                "1. 在浏览器中登录 B站（推荐）\n"
                "2. 设置环境变量 BILIBILI_SESSDATA"
            )
        self._sessdata = sessdata
        return sessdata

    def _get_cookies(self) -> dict:
        return {'SESSDATA': self._ensure_sessdata()}

    async def get_info(self, source: str) -> Dict[str, Any]:
        """获取 B站视频基本信息"""
        bvid = self._extract_bvid(source)
        url = f"{API_BASE_URL}/x/web-interface/view?bvid={bvid}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com/'
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, cookies=self._get_cookies())
            response.raise_for_status()
            data = response.json()

        if data['code'] != 0:
            raise ValueError(f"B站 API 返回错误: {data.get('message', '未知错误')}")

        video_data = data['data']
        return {
            "title": video_data.get('title'),
            "id": video_data.get('bvid'),
            "duration": video_data.get('duration', 0),
            "description": video_data.get('desc', ''),
            "author": video_data.get('owner', {}).get('name'),
            "has_subtitle": bool(video_data.get('subtitle', {}).get('list')),
            "cid": video_data.get('cid')
        }

    async def list_subtitles(self, source: str) -> Dict[str, Any]:
        """列出 B站视频可用的字幕"""
        video_info = await self.get_info(source)
        bvid = video_info['id']
        cid = video_info.get('cid')

        url = f"{API_BASE_URL}/x/player/wbi/v2?bvid={bvid}&cid={cid}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': f'https://www.bilibili.com/video/{bvid}',
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, cookies=self._get_cookies())
            response.raise_for_status()
            data = response.json()

        if data['code'] != 0:
            return {"available": False, "subtitles": [], "subtitle_count": 0, "error": data.get('message')}

        subtitle_list = data['data'].get('subtitle', {}).get('subtitles', [])
        subtitles = [
            {"lan": sub.get('lan'), "lan_doc": sub.get('lan_doc'), "subtitle_url": sub.get('subtitle_url')}
            for sub in subtitle_list
        ]

        return {"available": len(subtitles) > 0, "subtitles": subtitles, "subtitle_count": len(subtitles)}

    async def download_subtitle(
        self,
        source: str,
        format: ResponseFormat = ResponseFormat.TEXT,
        model_size: str = "large",
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """下载 B站视频字幕，无字幕时自动 ASR 兜底"""
        try:
            subtitle_info = await self.list_subtitles(source)

            if subtitle_info['available']:
                # 有 API 字幕
                zh_subtitle = None
                for sub in subtitle_info['subtitles']:
                    if sub['lan'] in ['ai-zh', 'zh-Hans', 'zh-CN', 'zh']:
                        zh_subtitle = sub
                        break
                if not zh_subtitle:
                    zh_subtitle = subtitle_info['subtitles'][0]

                video_info = await self.get_info(source)
                subtitle_url = zh_subtitle['subtitle_url']
                if not subtitle_url.startswith('http'):
                    subtitle_url = 'https:' + subtitle_url

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(subtitle_url)
                    response.raise_for_status()
                    subtitle_json = response.json()

                body = subtitle_json.get('body', [])
                log_success(f"API 获取成功，共 {len(body)} 条字幕")

                segments = [
                    {"start": item.get("from", 0), "end": item.get("to", 0), "content": item.get("content", "")}
                    for item in body
                ]

                return format_subtitle(segments, video_info['title'], format, source="bilibili_api")

            # 无 API 字幕，ASR 兜底
            log_warning("该视频没有可用字幕，切换到 ASR 模式")
            return await self._download_with_asr(source, format, model_size, show_progress)

        except Exception as e:
            return {"error": f"下载字幕失败: {type(e).__name__}", "message": str(e)}

    async def _download_with_asr(
        self,
        source: str,
        format: ResponseFormat,
        model_size: str,
        show_progress: bool
    ) -> Dict[str, Any]:
        """ASR 兜底下载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                log_step("下载视频并提取音频")
                audio_file, video_title, video_id = await self.download_and_extract_audio(
                    source, temp_dir, show_progress
                )
                log_success(f"音频提取完成: {os.path.basename(audio_file)}")

                log_step("ASR 语音识别", "这可能需要几分钟...")
                asr_result = await transcribe_with_asr(audio_file, model_size, show_progress)

                segments = asr_result.get("segments", [])
                log_success(f"ASR 完成，共 {len(segments)} 个片段")

                formatted = [
                    {"start": seg["start"], "end": seg["end"], "content": seg["text"]}
                    for seg in segments
                ]

                return format_subtitle(formatted, video_title, format, source="whisper_asr")

            except subprocess.CalledProcessError as e:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
                suggestion = "请检查网络连接"
                if 'not found' in stderr.lower():
                    suggestion = "请确保已安装 yt-dlp 和 ffmpeg"
                return {"error": f"ASR失败: {type(e).__name__}", "message": str(e), "suggestion": suggestion}
            except Exception as e:
                return {"error": f"ASR失败: {type(e).__name__}", "message": str(e)}

    async def download_video(
        self,
        source: str,
        output_dir: str,
        show_progress: bool = True
    ) -> tuple[str, str, str]:
        """下载 B站视频"""
        info = await self.get_info(source)
        video_title = info.get("title", "video")
        bvid = info.get("id", self._extract_bvid(source))

        safe_title = make_safe_filename(video_title)
        video_filename = os.path.join(output_dir, f"{safe_title}.mp4")

        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(video_filename):
            return video_filename, video_title, bvid

        if show_progress:
            log_step("正在下载视频")

        result = subprocess.run(['yt-dlp', '--quiet', '--no-progress', '-o', video_filename, f"https://www.bilibili.com/video/{bvid}"], capture_output=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, stderr=result.stderr.decode('utf-8', errors='ignore'))

        return video_filename, video_title, bvid

    def extract_audio(self, video_file: str, output_dir: Optional[str] = None, show_progress: bool = True) -> str:
        return extract_audio(video_file, output_dir, show_progress)
