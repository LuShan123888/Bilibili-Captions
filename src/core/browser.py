"""
浏览器 Cookie 读取模块
支持从 Arc、Chrome、Edge、Firefox、Brave 浏览器读取 B站 SESSDATA
"""

from http.cookiejar import MozillaCookieJar
import os
import shutil
import sqlite3
import sys
import tempfile
from typing import Optional
from pathlib import Path

BILIBILI_DOMAIN = ".bilibili.com"
ARC_PROFILE = Path("~/Library/Application Support/Arc/User Data/Default").expanduser()
ARC_BROWSER_DIR = ARC_PROFILE.parent


class QuietCookieLogger:
    """Suppress yt-dlp cookie extraction logs."""

    def debug(self, *args: object, **kwargs: object) -> None:
        pass

    def info(self, *args: object, **kwargs: object) -> None:
        pass

    def warning(self, *args: object, **kwargs: object) -> None:
        pass

    def error(self, *args: object, **kwargs: object) -> None:
        pass

    def progress_bar(self) -> None:
        return None


def _log_debug(message: str) -> None:
    """打印调试日志"""
    from .logging import log_debug as _ld

    _ld(f"[browser] {message}")


def _cookie_matches_domain(cookie_domain: str, domain_suffix: str) -> bool:
    domain = cookie_domain.lstrip(".").lower()
    suffix = domain_suffix.lstrip(".").lower()
    return domain == suffix or domain.endswith(f".{suffix}")


def load_arc_cookies(
    domain_suffix: Optional[str] = None,
    required_cookie_names: Optional[set[str]] = None,
    log: bool = True,
) -> Optional[MozillaCookieJar]:
    """读取 Arc Cookie，可按域名和 Cookie 名过滤。

    Arc 基于 Chromium，但 yt-dlp/browser-cookie3 不直接把它作为标准浏览器暴露。
    这里复用 yt-dlp 的 Chromium Cookie 解密能力，并临时把浏览器目录指向 Arc。
    """
    if not (ARC_PROFILE / "Cookies").exists():
        if log:
            _log_debug("Arc Cookie 数据库不存在")
        return None

    try:
        from yt_dlp import cookies as ytdlp_cookies
    except Exception as exc:
        if log:
            _log_debug(f"yt-dlp Python 模块不可用，无法读取 Arc Cookie: {exc}")
        return None

    original_settings = ytdlp_cookies._get_chromium_based_browser_settings

    def arc_settings(browser_name: str) -> dict[str, object]:
        return {
            "browser_dir": str(ARC_BROWSER_DIR),
            "keyring_name": "Arc",
            "supports_profiles": True,
        }

    try:
        ytdlp_cookies._get_chromium_based_browser_settings = arc_settings
        source_jar = ytdlp_cookies.extract_cookies_from_browser(
            "chrome",
            profile=str(ARC_PROFILE),
            logger=QuietCookieLogger(),
        )
    except Exception as exc:
        if log:
            _log_debug(f"Arc Cookie 读取失败: {type(exc).__name__}: {exc}")
        return None
    finally:
        ytdlp_cookies._get_chromium_based_browser_settings = original_settings

    filtered = MozillaCookieJar()
    matched_names: set[str] = set()
    for cookie in source_jar:
        if domain_suffix and not _cookie_matches_domain(cookie.domain, domain_suffix):
            continue
        filtered.set_cookie(cookie)
        matched_names.add(cookie.name)

    if len(filtered) == 0:
        if log:
            suffix = domain_suffix or "all domains"
            _log_debug(f"Arc 未找到匹配 Cookie: {suffix}")
        return None
    if required_cookie_names and not required_cookie_names.issubset(matched_names):
        if log:
            missing = ", ".join(sorted(required_cookie_names - matched_names))
            _log_debug(f"Arc Cookie 缺少必要字段: {missing}")
        return None
    return filtered


def export_arc_cookies(
    cookie_file: Path,
    domain_suffix: Optional[str] = None,
    required_cookie_names: Optional[set[str]] = None,
    log: bool = True,
) -> bool:
    """导出 Arc Cookie 为 Netscape cookie 文件，供 yt-dlp --cookies 使用。"""
    jar = load_arc_cookies(domain_suffix, required_cookie_names, log)
    if jar is None:
        return False

    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    jar.filename = str(cookie_file)
    jar.save(ignore_discard=True, ignore_expires=True)
    return True


def get_arc_cookie(log: bool = True) -> Optional[str]:
    """从 Arc 浏览器读取 B站 SESSDATA。"""
    jar = load_arc_cookies(
        BILIBILI_DOMAIN,
        required_cookie_names={"SESSDATA"},
        log=log,
    )
    if jar is None:
        return None

    for cookie in jar:
        if cookie.name == "SESSDATA":
            if log:
                _log_debug("从 Arc 找到 SESSDATA")
            return cookie.value
    return None


def _get_chromium_cookie_file(browser_name: str) -> Optional[Path]:
    """获取 Chromium 系浏览器的 Cookie 文件路径"""
    home = Path.home()

    browser_paths_map = {
        "chrome": "Library/Application Support/Google/Chrome",
        "edge": "Library/Application Support/Microsoft Edge",
        "brave": "Library/Application Support/BraveSoftware/Brave-Browser",
        "opera": "Library/Application Support/com.operasoftware.Opera",
    }

    if browser_name not in browser_paths_map:
        return None

    base_path = home / browser_paths_map[browser_name]
    if not base_path.exists():
        return None

    # 尝试不同的配置文件夹
    profile_names = ["Default", "Profile 1", "Profile 2"]
    for profile in profile_names:
        cookie_file = base_path / profile / "Cookies"
        if cookie_file.exists():
            return cookie_file

    return None


def _get_firefox_cookie_file() -> Optional[Path]:
    """获取 Firefox 的 Cookie 文件路径"""
    home = Path.home()
    firefox_base = home / "Library" / "Application Support" / "Firefox" / "Profiles"

    if not firefox_base.exists():
        return None

    for profile_dir in firefox_base.iterdir():
        if profile_dir.is_dir():
            cookie_file = profile_dir / "cookies.sqlite"
            if cookie_file.exists():
                return cookie_file

    return None


def _extract_sessdata_from_sqlite(cookie_file: Path) -> Optional[str]:
    """从 SQLite Cookie 数据库中提取 SESSDATA

    注意：Chromium 浏览器在 macOS 上的 Cookie 是加密的。
    直接读取 SQLite 只能获取加密数据，需要使用 browser-cookie3 库来解密。

    Args:
        cookie_file: Cookie 数据库文件路径

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    # Chrome/Edge 在使用时可能锁定数据库，需要复制到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
        temp_path = temp_file.name

    try:
        shutil.copy2(cookie_file, temp_path)

        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cookies'")
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return None

        # 查询 SESSDATA (Chromium 格式)
        cursor.execute("SELECT value FROM cookies WHERE host_key LIKE ?", (f"%{BILIBILI_DOMAIN}",))

        for row in cursor.fetchall():
            value = row[0]
            if value and len(value) > 10:
                # 注意：这里返回的可能是加密数据
                # 在 macOS 上，Chromium Cookie 使用系统 Keychain 加密
                return value

        return None

    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _try_browser_cookie3(log: bool = True) -> Optional[str]:
    """尝试使用 browser-cookie3 库读取加密 Cookie

    Args:
        log: 是否打印日志

    Returns:
        SESSDATA 字符串，未找到或库不可用返回 None
    """
    try:
        import browser_cookie3

        if log:
            _log_debug("browser-cookie3 库已加载")

        # 尝试不同的浏览器
        browsers_to_try = [
            ("Chrome", browser_cookie3.chrome),
            ("Edge", browser_cookie3.edge),
            ("Firefox", browser_cookie3.firefox),
            ("Brave", browser_cookie3.brave),
            ("Opera", browser_cookie3.opera),
        ]

        for browser_name, browser_func in browsers_to_try:
            try:
                if log:
                    _log_debug(f"尝试从 {browser_name} 读取...")
                # 抑制 browser_cookie3 内部的 tqdm 进度条输出
                with open(os.devnull, "w") as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    try:
                        cookie_jar = browser_func(domain_name=BILIBILI_DOMAIN)
                    finally:
                        sys.stderr = old_stderr
                for cookie in cookie_jar:
                    if cookie.name == "SESSDATA":
                        if log:
                            _log_debug(f"从 {browser_name} 找到 SESSDATA")
                        return cookie.value
                if log:
                    _log_debug(f"{browser_name} 未找到 SESSDATA")
            except Exception as e:
                if log:
                    _log_debug(f"{browser_name} 读取失败: {type(e).__name__}: {e}")
                continue

        return None

    except ImportError:
        if log:
            _log_debug("browser-cookie3 库未安装")
        return None
    except Exception as e:
        if log:
            _log_debug(f"browser-cookie3 错误: {type(e).__name__}: {e}")
        return None


def get_chrome_cookie(log: bool = True) -> Optional[str]:
    """从 Chrome 浏览器读取 SESSDATA

    Args:
        log: 是否打印日志

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    return _try_browser_cookie3(log)


def get_edge_cookie(log: bool = True) -> Optional[str]:
    """从 Edge 浏览器读取 SESSDATA

    Args:
        log: 是否打印日志

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    return _try_browser_cookie3(log)


def get_firefox_cookie(log: bool = True) -> Optional[str]:
    """从 Firefox 浏览器读取 SESSDATA

    Args:
        log: 是否打印日志

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    return _try_browser_cookie3(log)


def get_brave_cookie(log: bool = True) -> Optional[str]:
    """从 Brave 浏览器读取 SESSDATA

    Args:
        log: 是否打印日志

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    return _try_browser_cookie3(log)


# 记录最后成功读取的浏览器
_last_successful_browser: Optional[str] = None


def get_browser_name(browser: str = "arc") -> Optional[str]:
    """获取最后成功读取 Cookie 的浏览器名称

    Args:
        browser: 传入的浏览器参数（用于默认值）

    Returns:
        浏览器名称，如果未成功读取过则返回传入的参数
    """
    global _last_successful_browser
    return _last_successful_browser or browser


def get_sessdata_from_browser(browser: str = "arc", log: bool = True) -> Optional[str]:
    """从浏览器读取 SESSDATA

    Args:
        browser: 浏览器类型 ("auto", "arc", "chrome", "edge", "firefox", "brave")
                auto 模式会自动尝试所有浏览器
        log: 是否打印日志

    Returns:
        SESSDATA 字符串，未找到返回 None
    """
    global _last_successful_browser
    browsers = []

    if browser == "auto":
        browsers = ["arc", "chrome", "edge", "brave", "firefox"]
    else:
        browsers = [browser.lower()]

    for browser_name in browsers:
        if browser_name == "arc":
            sessdata = get_arc_cookie(log)
        elif browser_name == "chrome":
            sessdata = get_chrome_cookie(log)
        elif browser_name == "edge":
            sessdata = get_edge_cookie(log)
        elif browser_name == "firefox":
            sessdata = get_firefox_cookie(log)
        elif browser_name == "brave":
            sessdata = get_brave_cookie(log)
        else:
            continue

        if sessdata:
            _last_successful_browser = browser_name
            return sessdata

    return None


def list_available_browsers() -> list:
    """列出系统中可用的浏览器

    Returns:
        可用浏览器列表，如 ["arc", "chrome", "edge"]
    """
    home = Path.home()
    available = []

    # 检查 Arc
    if ARC_PROFILE.exists():
        available.append("arc")

    # 检查 Chrome
    chrome_path = home / "Library" / "Application Support" / "Google" / "Chrome"
    if chrome_path.exists():
        available.append("chrome")

    # 检查 Edge
    edge_path = home / "Library" / "Application Support" / "Microsoft Edge"
    if edge_path.exists():
        available.append("edge")

    # 检查 Firefox
    firefox_path = home / "Library" / "Application Support" / "Firefox" / "Profiles"
    if firefox_path.exists():
        available.append("firefox")

    # 检查 Brave
    brave_path = home / "Library" / "Application Support" / "BraveSoftware" / "Brave-Browser"
    if brave_path.exists():
        available.append("brave")

    return available
