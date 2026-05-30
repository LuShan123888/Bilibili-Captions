"""
测试用例 - YouTube 视频字幕下载

测试视频:
1. kQ-aFczITCg - 有 API 字幕
2. 5GJU5-UMNWk - 无字幕 (ASR 兜底)
"""

import asyncio
import pytest

from service import get_service, get_service_name
from core.formatter import ResponseFormat


YOUTUBE_WITH_SUBTITLES = "https://www.youtube.com/watch?v=kQ-aFczITCg"
YOUTUBE_WITHOUT_SUBTITLES = "https://www.youtube.com/watch?v=02U8vkSHyko"


@pytest.mark.asyncio
async def test_youtube_with_api_subtitles():
    """测试有 API 字幕的 YouTube 视频"""
    service = get_service(YOUTUBE_WITH_SUBTITLES)

    info = await service.get_info(YOUTUBE_WITH_SUBTITLES)
    assert info["id"] == "kQ-aFczITCg"

    result = await service.download_subtitle(YOUTUBE_WITH_SUBTITLES, ResponseFormat.TEXT)
    assert "error" not in result
    assert result["source"] == "youtube_api"
    assert result["subtitle_count"] > 0


@pytest.mark.asyncio
async def test_youtube_with_asr_fallback():
    """测试无字幕 YouTube 视频 ASR 兜底"""
    service = get_service(YOUTUBE_WITHOUT_SUBTITLES)

    result = await service.download_subtitle(YOUTUBE_WITHOUT_SUBTITLES, ResponseFormat.TEXT, model_size="base")
    assert "error" not in result
    assert result["source"] == "whisper_asr"
    assert result["subtitle_count"] > 0


@pytest.mark.asyncio
async def test_platform_detection():
    """测试平台识别"""
    assert get_service_name(YOUTUBE_WITH_SUBTITLES) == "youtube"
    assert get_service_name("https://youtu.be/kQ-aFczITCg") == "youtube"


if __name__ == "__main__":
    async def run():
        print("\n=== 测试 1: 有 API 字幕 ===")
        s1 = get_service(YOUTUBE_WITH_SUBTITLES)
        r1 = await s1.download_subtitle(YOUTUBE_WITH_SUBTITLES, ResponseFormat.TEXT)
        print(f"来源: {r1.get('source')}, 字幕数: {r1.get('subtitle_count')}")

        print("\n=== 测试 2: 无字幕 ASR 兜底 ===")
        s2 = get_service(YOUTUBE_WITHOUT_SUBTITLES)
        r2 = await s2.download_subtitle(YOUTUBE_WITHOUT_SUBTITLES, ResponseFormat.TEXT, model_size="base")
        print(f"来源: {r2.get('source')}, 字幕数: {r2.get('subtitle_count')}")
        print("\n✓ 完成")

    asyncio.run(run())
