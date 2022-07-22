import typer

NAME = "Meddocan"
HELP = "Meddocan Command-Line Interface"
app = typer.Typer(name=NAME, help=HELP)

# Wrappers for Typer's annotations.
Arg = typer.Argument
Opt = typer.Option
