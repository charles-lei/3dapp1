"""外部命令执行封装。所有 nerfstudio/COLMAP 调用都经由此处，便于测试注入。"""

import subprocess
from collections.abc import Callable

from app.errors import CommandError


def run_command(cmd: list[str], on_line: Callable[[str], None] | None = None) -> str:
    """运行命令，合并 stdout/stderr 逐行回调，返回完整输出。非零退出抛 CommandError。"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        output_lines.append(line)
        if on_line is not None:
            on_line(line)
    returncode = proc.wait()
    output = "\n".join(output_lines)
    if returncode != 0:
        raise CommandError(cmd, returncode, output)
    return output
