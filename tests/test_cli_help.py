import sys

import pytest

from handler.cli import main


def test_help_describes_cli_contract(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["video-captions", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.err == ""
    assert "优先获取平台字幕" in captured.out
    assert "--browser 仅用于 B站或 YouTube" in captured.out
    assert "Agent/脚本建议使用 --format json" in captured.out
    assert "字幕内容写入 stdout" in captured.out
    assert "日志/错误  写入 stderr" in captured.out
    assert "失败退出码非 0" in captured.out
