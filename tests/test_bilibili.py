"""
测试用例 - B站视频字幕下载

测试视频:
1. BV16YC3BrEDz - 有 API 字幕
2. BV1qViQBwELr - 无字幕 (ASR 兜底)
"""

import asyncio
import pytest

from service import get_service
from service.bilibili import BilibiliService
from core.cookie import get_sessdata
from core.formatter import ResponseFormat


def require_sessdata():
    result = get_sessdata()
    if not result:
        pytest.skip("需要 SESSDATA")
    return result


VIDEO_WITH_SUBTITLES = "https://www.bilibili.com/video/BV16YC3BrEDz/"
VIDEO_WITHOUT_SUBTITLES = "https://www.bilibili.com/video/BV1qViQBwELr/"
WATCHLATER_VIDEO = (
    "https://www.bilibili.com/list/watchlater/?bvid=BV1eXVh6LEwN"
    "&oid=116656814823166"
    "&watchlater_cfg=%7B%22viewed%22:0,%22key%22:%22%22,%22asc%22:false%7D"
    "&vd_source=d232d8c25800736106d48d1cd29856a7"
)


def test_extract_bvid_from_watchlater_url():
    """watchlater 链接应优先使用 URL 中固定的 bvid 参数"""
    service = BilibiliService()

    assert service.is_supported(WATCHLATER_VIDEO)
    assert service._extract_bvid(WATCHLATER_VIDEO) == "BV1eXVh6LEwN"


def test_extract_bvid_from_video_url_with_query():
    """普通视频页带查询参数时仍只提取 BV 号"""
    service = BilibiliService()

    assert (
        service._extract_bvid("https://www.bilibili.com/video/BV16YC3BrEDz?spm_id_from=333.999")
        == "BV16YC3BrEDz"
    )


def test_extract_bvid_prefers_video_path_over_query():
    """普通视频页优先使用路径中的 BV 号"""
    service = BilibiliService()

    assert (
        service._extract_bvid("https://www.bilibili.com/video/BV16YC3BrEDz?bvid=BV1eXVh6LEwN")
        == "BV16YC3BrEDz"
    )


@pytest.mark.asyncio
async def test_video_with_api_subtitles():
    """测试有 API 字幕的视频"""
    require_sessdata()
    service = get_service(VIDEO_WITH_SUBTITLES)

    info = await service.get_info(VIDEO_WITH_SUBTITLES)
    assert info["id"] == "BV16YC3BrEDz"

    result = await service.download_subtitle(VIDEO_WITH_SUBTITLES, ResponseFormat.TEXT)
    assert "error" not in result
    assert result["source"] == "bilibili_api"
    assert result["subtitle_count"] > 180


@pytest.mark.asyncio
async def test_video_with_asr_fallback():
    """测试无字幕视频 ASR 兜底"""
    require_sessdata()
    service = get_service(VIDEO_WITHOUT_SUBTITLES)

    result = await service.download_subtitle(VIDEO_WITHOUT_SUBTITLES, ResponseFormat.TEXT, model_size="base")
    assert "error" not in result
    assert result["source"] == "whisper_asr"
    assert result["subtitle_count"] > 0


if __name__ == "__main__":
    async def run():
        require_sessdata()

        print("\n=== 测试 1: 有 API 字幕 ===")
        s1 = get_service(VIDEO_WITH_SUBTITLES)
        r1 = await s1.download_subtitle(VIDEO_WITH_SUBTITLES, ResponseFormat.TEXT)
        print(f"来源: {r1.get('source')}, 字幕数: {r1.get('subtitle_count')}")

        print("\n=== 测试 2: 无字幕 ASR 兜底 ===")
        s2 = get_service(VIDEO_WITHOUT_SUBTITLES)
        r2 = await s2.download_subtitle(VIDEO_WITHOUT_SUBTITLES, ResponseFormat.TEXT, model_size="base")
        print(f"来源: {r2.get('source')}, 字幕数: {r2.get('subtitle_count')}")
        print("\n✓ 完成")

    asyncio.run(run())
