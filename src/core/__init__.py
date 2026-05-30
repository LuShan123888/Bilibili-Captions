"""
Core 层 - 通用能力模块

提供日志、文本处理、音频处理、ASR、Cookie 读取、字幕格式化等通用功能
"""

from .logging import (
    set_verbose_log,
    log_info,
    log_step,
    log_success,
    log_warning,
    log_error,
    log_debug,
)
from .text import make_safe_filename, convert_to_simplified
from .audio import extract_audio, is_video_file, is_audio_file
from .asr import transcribe_with_asr
from .cookie import get_sessdata, get_sessdata_with_source, require_sessdata
from .formatter import format_subtitle, ResponseFormat
from .browser import (
    export_arc_cookies,
    get_sessdata_from_browser,
    get_browser_name,
    list_available_browsers,
    load_arc_cookies,
)

__all__ = [
    # Logging
    "set_verbose_log",
    "log_info",
    "log_step",
    "log_success",
    "log_warning",
    "log_error",
    "log_debug",
    # Text
    "make_safe_filename",
    "convert_to_simplified",
    # Audio
    "extract_audio",
    "is_video_file",
    "is_audio_file",
    # ASR
    "transcribe_with_asr",
    # Cookie
    "get_sessdata",
    "get_sessdata_with_source",
    "require_sessdata",
    # Browser
    "export_arc_cookies",
    "get_sessdata_from_browser",
    "get_browser_name",
    "list_available_browsers",
    "load_arc_cookies",
    # Formatter
    "format_subtitle",
    "ResponseFormat",
]
