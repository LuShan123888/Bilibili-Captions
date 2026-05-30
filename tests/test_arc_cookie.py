from pathlib import Path

from core import browser as browser_module
from core.browser import get_browser_name, get_sessdata_from_browser, list_available_browsers
from core.cookie import get_sessdata
from core.ytdlp_cookie import ytdlp_cookie_args


def test_default_browser_uses_arc_for_sessdata(monkeypatch):
    browser_module._last_successful_browser = None
    calls = []

    def fake_arc_cookie(log=True):
        calls.append("arc")
        return "arc-sessdata"

    monkeypatch.setattr(browser_module, "get_arc_cookie", fake_arc_cookie)

    assert get_sessdata_from_browser(log=False) == "arc-sessdata"
    assert get_sessdata() == "arc-sessdata"
    assert calls == ["arc", "arc"]
    assert get_browser_name() == "arc"

    browser_module._last_successful_browser = None


def test_auto_prefers_arc_for_sessdata(monkeypatch):
    browser_module._last_successful_browser = None
    calls = []

    def fake_arc_cookie(log=True):
        calls.append("arc")
        return "arc-sessdata"

    monkeypatch.setattr(browser_module, "get_arc_cookie", fake_arc_cookie)

    assert get_sessdata_from_browser("auto", log=False) == "arc-sessdata"
    assert calls == ["arc"]

    browser_module._last_successful_browser = None


def test_arc_browser_can_be_listed(monkeypatch, tmp_path):
    arc_profile = tmp_path / "Arc" / "User Data" / "Default"
    arc_profile.mkdir(parents=True)
    monkeypatch.setattr(browser_module, "ARC_PROFILE", arc_profile)

    assert "arc" in list_available_browsers()


def test_ytdlp_cookie_args_exports_arc_cookie_file(monkeypatch, tmp_path):
    exported = []

    def fake_export(cookie_file, domain_suffix=None, required_cookie_names=None, log=True):
        exported.append((cookie_file, domain_suffix, required_cookie_names))
        Path(cookie_file).write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        return True

    monkeypatch.setattr("core.ytdlp_cookie.export_arc_cookies", fake_export)

    with ytdlp_cookie_args(
        "arc",
        domain_suffix="bilibili.com",
        required_cookie_names={"SESSDATA"},
        temp_dir=str(tmp_path),
    ) as args:
        assert args == ["--cookies", str(tmp_path / "arc-cookies.txt")]

    assert exported == [(tmp_path / "arc-cookies.txt", "bilibili.com", {"SESSDATA"})]


def test_ytdlp_cookie_args_uses_standard_browser_for_non_arc():
    with ytdlp_cookie_args("chrome") as args:
        assert args == ["--cookies-from-browser", "chrome"]


def test_ytdlp_cookie_args_defaults_to_arc(monkeypatch, tmp_path):
    def fake_named_tempfile(delete=False, suffix=""):
        class Handle:
            name = str(tmp_path / f"default{suffix}")

            def close(self):
                pass

        return Handle()

    def fake_export(cookie_file, domain_suffix=None, required_cookie_names=None, log=True):
        Path(cookie_file).write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        return True

    monkeypatch.setattr("core.ytdlp_cookie.tempfile.NamedTemporaryFile", fake_named_tempfile)
    monkeypatch.setattr("core.ytdlp_cookie.export_arc_cookies", fake_export)

    with ytdlp_cookie_args(None) as args:
        assert args == ["--cookies", str(tmp_path / "default-arc-cookies.txt")]
