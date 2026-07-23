import typer

from rl_algorithms.algorithms import train_trpo

app = typer.Typer(help="RL Algorithms main.py")
app.command(name="trpo")(train_trpo)

if __name__ == "__main__":
    app()