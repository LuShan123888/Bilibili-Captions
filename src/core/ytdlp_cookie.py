"""Cookie helpers for yt-dlp subprocess calls."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator, Optional

from .browser import export_arc_cookies


@contextmanager
def ytdlp_cookie_args(
    browser: Optional[str],
    *,
    domain_suffix: Optional[str] = None,
    required_cookie_names: Optional[set[str]] = None,
    temp_dir: Optional[str] = None,
) -> Iterator[list[str]]:
    """Return yt-dlp cookie arguments, exporting Arc cookies when needed."""
    browser_name = (browser or "arc").lower()

    if browser_name in {"arc", "auto"}:
        cleanup_path: Optional[Path] = None
        if temp_dir:
            cookie_file = Path(temp_dir) / "arc-cookies.txt"
        else:
            handle = tempfile.NamedTemporaryFile(delete=False, suffix="-arc-cookies.txt")
            handle.close()
            cookie_file = Path(handle.name)
            cleanup_path = cookie_file

        if export_arc_cookies(cookie_file, domain_suffix, required_cookie_names, log=False):
            try:
                yield ["--cookies", str(cookie_file)]
            finally:
                if cleanup_path:
                    cleanup_path.unlink(missing_ok=True)
            return

        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)

    if browser_name not in {"auto", "arc"}:
        yield ["--cookies-from-browser", browser_name]
        return

    yield []
