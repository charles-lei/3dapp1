"""结构化错误定义。code 取值见 spec §4 错误码集合。"""


class PipelineError(Exception):
    """管线业务错误，code 会透传给 App。"""

    def __init__(self, code: str, message_zh: str):
        super().__init__(f"{code}: {message_zh}")
        self.code = code
        self.message_zh = message_zh


class CommandError(Exception):
    """外部命令（COLMAP/nerfstudio）执行失败。"""

    def __init__(self, cmd: list[str], returncode: int, output: str):
        self.cmd = cmd
        self.returncode = returncode
        self.output = output
        super().__init__(f"command failed ({returncode}): {' '.join(cmd)}\n{output[-2000:]}")
