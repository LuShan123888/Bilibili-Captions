"""
Cookie 读取 - 从浏览器读取 B站 SESSDATA
"""

import os
from typing import Optional

from .logging import log_step, log_success, log_warning, log_debug


def get_sessdata(browser: Optional[str] = "arc") -> Optional[str]:
    """获取 SESSDATA，按优先级从浏览器、环境变量读取

    Args:
        browser: 浏览器类型 ("arc", "auto", "chrome", "edge", "firefox", "brave")

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    result, _ = get_sessdata_with_source(browser, log=False)
    return result


def get_sessdata_with_source(
    browser: Optional[str] = "arc",
    log: bool = True
) -> tuple[Optional[str], Optional[str]]:
    """获取 SESSDATA 及其来源

    Args:
        browser: 浏览器类型
        log: 是否打印日志

    Returns:
        (SESSDATA, 来源) 元组
    """
    if log:
        log_step("获取 SESSDATA")

    # 1. 从浏览器读取
    if browser is not False:
        if log:
            log_debug(f"尝试从浏览器读取 (模式: {browser or 'auto'})...")

        from .browser import get_sessdata_from_browser, get_browser_name
        browser_sessdata = get_sessdata_from_browser(browser or "auto", log=log)
        if browser_sessdata:
            detected_browser = get_browser_name(browser or "auto")
            if log:
                log_success(f"从浏览器获取 SESSDATA ({detected_browser})")
            return browser_sessdata, f"browser:{detected_browser}"
        if log:
            log_debug("浏览器未找到 SESSDATA")
    else:
        if log:
            log_debug("浏览器读取已禁用")

    # 2. 从环境变量读取
    if log:
        log_debug("检查环境变量 BILIBILI_SESSDATA...")
    env_sessdata = os.environ.get('BILIBILI_SESSDATA')
    if env_sessdata:
        if log:
            log_success("从环境变量获取 SESSDATA")
        return env_sessdata, "env"
    if log:
        log_debug("环境变量未设置")

    if log:
        log_warning("未获取到 SESSDATA")
    return None, None


def require_sessdata(browser: Optional[str] = "arc") -> str:
    """获取 SESSDATA，如果没有找到则抛出异常

    Args:
        browser: 浏览器类型

    Returns:
        SESSDATA 字符串

    Raises:
        ValueError: 未找到 SESSDATA
    """
    result = get_sessdata(browser)
    if not result:
        raise ValueError(
            "未找到 B站 SESSDATA 认证信息。请通过以下方式之一提供：\n"
            "1. 在浏览器中登录 B站（推荐，默认会自动读取）\n"
            "2. 设置环境变量 BILIBILI_SESSDATA"
        )
    return result
