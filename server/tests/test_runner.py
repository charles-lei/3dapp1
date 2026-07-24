import sys

import pytest

from app.errors import CommandError
from app.runner import run_command


def test_run_command_returns_output():
    out = run_command([sys.executable, "-c", "print('hello')"])
    assert "hello" in out


def test_run_command_streams_lines():
    lines = []
    run_command(
        [sys.executable, "-c", "print('a'); print('b')"],
        on_line=lines.append,
    )
    assert lines == ["a", "b"]


def test_run_command_raises_on_nonzero_exit():
    with pytest.raises(CommandError) as exc_info:
        run_command([sys.executable, "-c", "import sys; print('oops'); sys.exit(3)"])
    assert exc_info.value.returncode == 3
    assert "oops" in exc_info.value.output


def test_run_command_captures_stderr():
    out = run_command([sys.executable, "-c", "import sys; sys.stderr.write('err-line\\n')"])
    assert "err-line" in out
