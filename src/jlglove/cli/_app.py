import click
from rats import apps, cli
import pytorch_lightning as pl

from jlglove import rep


class Application(apps.Container, cli.Container, apps.PluginMixin):
    def execute(self) -> None:
        cli.create_group(click.Group(), self)()

    @cli.command()
    @click.option("--output-path", help="path to store generated training data")
    def _generate(self, output_path: str) -> None:
        """Create a directory of generated synthetic data."""
        pl.seed_everything(1)  # Set seed for reproducibility
        print(f"generating synthetic data into: {output_path}")

        sdg = rep.SyntheticDataGenerator()
        sdg.generate()

    @cli.command()
    @click.option("--input-path", help="path to raw tcr data")
    @click.option("--output-path", help="path to store generated training data")
    def _prepare(self, input_path: str, output_path: str) -> None:
        """Prepare train/test data from a directory of input tcr data."""
        print(f"processing data from: {input_path}")
        print(f"storing prepared data into: {output_path}")

    @cli.command()
    @click.option("--input-path", help="path to training data directory")
    @click.option("--output-path", help="path to store generated training data")
    def _train(self, input_path: str, output_path: str) -> None:
        """Train model embeddings from a prepared directory of tcr data."""
        print(f"training model from: {input_path}")
        print(f"storing embeddings into: {output_path}")

    @cli.command()
    @click.option("--model-path", help="path to trained model embeddings")
    @click.option("--input-path", help="path to unlabeled data")
    @click.option("--output-path", help="path to store predicted labels")
    def _infer(self, model_path: str, input_path: str, output_path: str) -> None:
        """Load trained model embeddings and run inference on a directory of unlabeled data."""
        print(f"loading model embeddings from: {model_path}")
        print(f"running inference on input data: {model_path}")
        print(f"storing predicted labels into: {output_path}")


def main() -> None:
    apps.run_plugin(Application)
