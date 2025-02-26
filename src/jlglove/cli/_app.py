import click
from rats import apps, cli


class Application(apps.Container, cli.Container, apps.PluginMixin):

    def execute(self) -> None:
        cli.create_group(click.Group(), self)()

    @cli.command()
    def prepare(self) -> None:
        print("preparing data")

    @cli.command()
    def train(self) -> None:
        print("training model")

    @cli.command()
    def infer(self) -> None:
        print("infer labels")


def main() -> None:
    apps.run_plugin(Application)
