from __future__ import annotations

from io import StringIO
from contextlib import redirect_stdout

from open_xtract.main import main


def test_cli_main_prints_message() -> None:
    buf = StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    assert "open-xtract: CLI not configured" in out

