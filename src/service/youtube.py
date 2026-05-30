"""
YouTube 服务 - 字幕下载和处理
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Dict, Any, Optional, List

from .base import SubtitleService
from core.formatter import ResponseFormat, format_subtitle
from core.audio import extract_audio
from core.asr import transcribe_with_asr
from core.logging import log_success, log_warning, log_step
from core.text import make_safe_filename
from core.ytdlp_cookie import ytdlp_cookie_args


YOUTUBE_LANG_PRIORITY = [
    "zh-Hans-en", "zh-Hant-en", "zh-Hans", "zh-Hant", "zh-CN", "zh-TW", "zh-HK", "zh", "en"
]


class YouTubeService(SubtitleService):
    """YouTube 字幕服务"""

    def __init__(self, browser: Optional[str] = "arc"):
        self.browser = browser

    @property
    def name(self) -> str:
        return "youtube"

    def is_supported(self, source: str) -> bool:
        patterns = [
            r'youtube\.com/watch', r'youtube\.com/shorts/', r'youtu\.be/',
            r'youtube\.com/embed/', r'youtube\.com/v/',
        ]
        for pattern in patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return True
        return False

    def _extract_video_id(self, url: str) -> str:
        if 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0].split('/')[0]
        if 'watch?v=' in url:
            return url.split('watch?v=')[1].split('&')[0].split('#')[0]
        if '/shorts/' in url:
            return url.split('/shorts/')[1].split('?')[0].split('/')[0]
        if '/embed/' in url:
            return url.split('/embed/')[1].split('?')[0].split('/')[0]
        if '/v/' in url:
            return url.split('/v/')[1].split('?')[0].split('/')[0]
        raise ValueError(f"无法从 URL 中提取 YouTube 视频 ID: {url}")

    def _get_cookie_args(self) -> List[str]:
        if self.browser and self.browser == "arc":
            return []
        if self.browser and self.browser != "auto":
            return ['--cookies-from-browser', self.browser]
        return []

    async def get_info(self, source: str) -> Dict[str, Any]:
        video_id = self._extract_video_id(source)
        with ytdlp_cookie_args(self.browser) as cookie_args:
            cmd = ['yt-dlp', '--quiet', '--no-progress', '--dump-json', '--no-download'] + cookie_args + [source]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr or str(e)
                if 'Sign in' in error_msg or 'age' in error_msg.lower():
                    raise ValueError("YouTube 视频需要登录")
                raise ValueError(f"获取视频信息失败: {error_msg[:200]}")
            info = json.loads(result.stdout)

        subtitles = list(info.get('subtitles', {}).keys())
        for lang in info.get('automatic_captions', {}).keys():
            if lang not in subtitles:
                subtitles.append(lang)

        return {
            "title": info.get('title', ''),
            "id": video_id,
            "duration": info.get('duration', 0),
            "description": info.get('description', ''),
            "author": info.get('uploader', info.get('channel', '')),
            "has_subtitle": len(subtitles) > 0,
            "available_subtitles": subtitles
        }

    async def list_subtitles(self, source: str) -> Dict[str, Any]:
        try:
            info = await self.get_info(source)
            langs = info.get('available_subtitles', [])
            subtitles = [{"lan": lang, "lan_doc": lang} for lang in langs]
            return {"available": len(subtitles) > 0, "subtitles": subtitles, "subtitle_count": len(subtitles)}
        except Exception as e:
            return {"available": False, "subtitles": [], "subtitle_count": 0, "error": str(e)}

    def _select_lang(self, available: List[str]) -> Optional[str]:
        for lang in YOUTUBE_LANG_PRIORITY:
            if lang in available:
                return lang
        for lang in available:
            if lang.startswith('zh'):
                return lang
        return available[0] if available else None

    async def download_subtitle(
        self,
        source: str,
        format: ResponseFormat = ResponseFormat.TEXT,
        model_size: str = "large",
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """下载 YouTube 视频字幕，无字幕时自动 ASR 兜底"""
        try:
            info = await self.get_info(source)
            available = info.get('available_subtitles', [])

            if available:
                lang = self._select_lang(available)
                if not lang:
                    return {"error": "无法选择字幕"}

                with tempfile.TemporaryDirectory() as temp_dir:
                    output = os.path.join(temp_dir, '%(id)s')
                    with ytdlp_cookie_args(self.browser, temp_dir=temp_dir) as cookie_args:
                        cmd = [
                            'yt-dlp', '--quiet', '--no-progress', '--write-subs',
                            '--write-auto-subs', '--sub-lang', lang, '--skip-download',
                            '--sub-format', 'json3', '-o', output
                        ] + cookie_args + [source]

                        subprocess.run(cmd, capture_output=True, text=True)

                    sub_file = None
                    for f in os.listdir(temp_dir):
                        if f.endswith('.json3') or f.endswith('.json'):
                            sub_file = os.path.join(temp_dir, f)
                            break

                    if sub_file:
                        with open(sub_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        segments = self._parse_json3(content)
                        if segments:
                            log_success(f"YouTube 字幕获取成功，共 {len(segments)} 条")
                            return format_subtitle(segments, info['title'], format, source="youtube_api")

            # 无字幕，ASR 兜底
            log_warning("该视频没有可用字幕，切换到 ASR 模式")
            return await self._download_with_asr(source, format, model_size, show_progress)

        except Exception as e:
            return {"error": f"下载字幕失败: {type(e).__name__}", "message": str(e)}

    def _parse_json3(self, content: str) -> Optional[List[Dict]]:
        try:
            data = json.loads(content)
            segments = []
            for event in data.get('events', []):
                text = ''.join(s.get('utf8', '') for s in event.get('segs', [])) if 'segs' in event else event.get('text', '')
                if text.strip():
                    segments.append({
                        'start': event.get('tStartMs', 0) / 1000.0,
                        'end': (event.get('tStartMs', 0) + event.get('dDurationMs', 0)) / 1000.0,
                        'content': text.strip()
                    })
            return segments
        except Exception:
            return None

    async def _download_with_asr(self, source: str, format: ResponseFormat, model_size: str, show_progress: bool) -> Dict[str, Any]:
        """ASR 兜底下载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                log_step("下载视频并提取音频")
                audio_file, video_title, video_id = await self.download_and_extract_audio(source, temp_dir, show_progress)
                log_success(f"音频提取完成: {os.path.basename(audio_file)}")

                log_step("ASR 语音识别", "这可能需要几分钟...")
                asr_result = await transcribe_with_asr(audio_file, model_size, show_progress)

                segments = asr_result.get("segments", [])
                log_success(f"ASR 完成，共 {len(segments)} 个片段")

                formatted = [{"start": s["start"], "end": s["end"], "content": s["text"]} for s in segments]
                return format_subtitle(formatted, video_title, format, source="whisper_asr")

            except subprocess.CalledProcessError as e:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
                return {"error": "ASR失败", "message": stderr[:200] or str(e)}
            except Exception as e:
                return {"error": f"ASR失败: {type(e).__name__}", "message": str(e)}

    async def download_video(self, source: str, output_dir: str, show_progress: bool = True) -> tuple[str, str, str]:
        info = await self.get_info(source)
        title = make_safe_filename(info.get("title", "video"))
        video_id = info.get("id", self._extract_video_id(source))
        filename = os.path.join(output_dir, f"{title}.mp4")

        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(filename):
            return filename, info.get("title", "video"), video_id

        if show_progress:
            log_step("正在下载 YouTube 视频")

        with ytdlp_cookie_args(self.browser) as cookie_args:
            cmd = [
                'yt-dlp', '--quiet', '--no-progress', '-o', filename,
                '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '--merge-output-format', 'mp4',
            ] + cookie_args + [source]

            result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr.decode('utf-8', errors='ignore'))

        return filename, info.get("title", "video"), video_id

    def extract_audio(self, video_file: str, output_dir: Optional[str] = None, show_progress: bool = True) -> str:
        return extract_audio(video_file, output_dir, show_progress)
