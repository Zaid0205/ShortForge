"""Allow `python -m shortforge "topic"` as an alternative to the `shortforge` command."""

from shortforge.cli import app

app(prog_name="shortforge")
