"""
Service 层 - 业务服务模块

提供服务工厂和各平台的字幕服务实现
"""

import os
from typing import Optional, Dict, Type

from .base import SubtitleService
from .local import LocalService
from .bilibili import BilibiliService
from .youtube import YouTubeService

# 服务注册表
_SERVICE_REGISTRY: Dict[str, Type[SubtitleService]] = {
    "local": LocalService,
    "bilibili": BilibiliService,
    "youtube": YouTubeService,
}


def get_service(
    source: str,
    browser: Optional[str] = "arc"
) -> Optional[SubtitleService]:
    """根据来源获取对应的服务实例

    Args:
        source: 视频来源（URL 或本地文件路径）
        browser: 指定从哪个浏览器读取 Cookie

    Returns:
        服务实例，如果无法识别来源则返回 None
    """
    # 首先检查是否为本地文件
    local_service = LocalService()
    if local_service.is_supported(source):
        return local_service

    # 检查 B站
    bilibili_service = BilibiliService(browser)
    if bilibili_service.is_supported(source):
        return bilibili_service

    # 检查 YouTube
    youtube_service = YouTubeService(browser)
    if youtube_service.is_supported(source):
        return youtube_service

    return None


def get_service_by_name(
    name: str,
    browser: Optional[str] = "arc"
) -> Optional[SubtitleService]:
    """根据服务名称获取服务实例

    Args:
        name: 服务名称 (local/bilibili/youtube)
        browser: 指定从哪个浏览器读取 Cookie

    Returns:
        服务实例
    """
    service_class = _SERVICE_REGISTRY.get(name)
    if service_class:
        if name == "local":
            return service_class()
        else:
            return service_class(browser)
    return None


def get_service_name(source: str) -> Optional[str]:
    """获取来源对应的服务名称

    Args:
        source: 视频来源

    Returns:
        服务名称，如果无法识别则返回 None
    """
    # 检查本地文件
    if os.path.exists(source):
        local_service = LocalService()
        if local_service.is_supported(source):
            return "local"

    # 检查 B站
    bilibili_service = BilibiliService()
    if bilibili_service.is_supported(source):
        return "bilibili"

    # 检查 YouTube
    youtube_service = YouTubeService()
    if youtube_service.is_supported(source):
        return "youtube"

    return None


def is_supported_source(source: str) -> bool:
    """检查来源是否被支持

    Args:
        source: 视频来源

    Returns:
        是否支持该来源
    """
    return get_service_name(source) is not None


__all__ = [
    # Base
    "SubtitleService",
    # Services
    "LocalService",
    "BilibiliService",
    "YouTubeService",
    # Factory functions
    "get_service",
    "get_service_by_name",
    "get_service_name",
    "is_supported_source",
]
