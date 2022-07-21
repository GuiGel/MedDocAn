import shlex

from typer.testing import CliRunner

from meddocan.cli import app

runner = CliRunner()


def meddocan_cli(command_string: str) -> str:
    command_list = shlex.split(command_string)
    result = runner.invoke(app, command_list)
    output = result.stdout.rstrip()
    return output
