from datetime import timedelta
from pathlib import Path

import click
import dask.dataframe as dd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from rats import apps, cli, logs

from jlglove import rep


class Application(apps.Container, cli.Container, apps.PluginMixin):
    def execute(self) -> None:
        cli.create_group(click.Group(), self)()

    @cli.command()
    @click.option("-n", type=int, default=5000, required=True)
    @click.option("-j", type=int, default=500, required=True)
    @click.option("-k", type=int, default=3, required=True)
    @click.option(
        "--output-path",
        default=".tmp/generated",
        help="path to store generated training data",
    )
    def _generate(self, n: int, j: int, k: int, output_path: str) -> None:
        """Create a directory of generated synthetic data."""
        pl.seed_everything(1)  # Set seed for reproducibility
        print(f"generating synthetic data into: {output_path}")

        sdg = rep.SyntheticDataGenerator(
            n=n,
            j=j,
            k=k,
            save_path=Path(output_path),
        )
        sdg.generate()

    @cli.command()
    @click.option("--input-path", help="path to raw tcr data")
    @click.option("--output-path", help="path to store generated training data")
    def _prepare(self, input_path: str, output_path: str) -> None:
        """Prepare train/test data from a directory of input tcr data."""
        print(f"processing data from: {input_path}")
        print(f"storing prepared data into: {output_path}")

    @cli.command()
    @click.option("--input-path", default=".tmp/generated", help="path to training data directory")
    @click.option(
        "--output-path",
        default=".tmp/results",
        help="path to store generated training data",
    )
    def _train(self, input_path: str, output_path: str) -> None:
        """Train model embeddings from a prepared directory of tcr data."""
        print(f"training model from: {input_path}")
        print(f"storing embeddings into: {output_path}")
        training_data_uri = Path(input_path) / "Glove_Synthentic_Data_500_t2.parquet"
        training_bioids_uri = (
            Path(input_path) / "Glove_Synthentic_Data_500_BioId_with_ES_JL.parquet"
        )

        num_tcr = 500
        emb_size = 100
        eval_epoch = 10
        num_epochs = 50
        ckpt_path = None
        train_partition_prop = 1.0
        batch_size = 1
        number_of_workers = 0
        jl_init = True
        l1_lambda = 0.0
        learning_rate = 0.05

        training_bioids = dd.read_parquet(path=str(training_bioids_uri)).compute()
        training_data = dd.read_parquet(path=str(training_data_uri))

        print(training_bioids.head())
        print(training_data.head())

        epoch_duration_printer = rep.EpochDurationPrinter(bioid_df=training_bioids)

        embedding_plotter = rep.EmbeddingPlotterCallback(
            bioid_df=training_bioids,
            eval_epochs=eval_epoch,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=output_path,
            filename="model-{epoch:02d}-{train_loss:.2f}",
            every_n_epochs=1,
            auto_insert_metric_name=True,
            save_weights_only=False,
            save_top_k=1,  # Save the top k checkpoints
        )

        custom_callbacks = [
            epoch_duration_printer,
            embedding_plotter,
            checkpoint_callback,
        ]

        custom_timeout = timedelta(minutes=120)
        DDPStrategy(
            find_unused_parameters=False,
            timeout=custom_timeout,
            process_group_backend="nccl",
            checkpoint_io=rep.CustomTorchCheckpointIO(),
        )  # TODO NCCL fails when gpus > 2; gloo

        devices = torch.cuda.device_count()
        for i in range(devices):
            print(f"device {i}:", torch.tensor(1, device=f"cuda:{i}").shape)

        # TODO: need to switch loggers
        # wandb_logger = WandbLogger(project="jl-glove", log_model="all")
        torch_logger = CSVLogger(save_dir=".tmp/torch-logs")
        if torch.cuda.is_available() and devices > 0:
            trainer = Trainer(
                max_epochs=num_epochs,
                logger=torch_logger,
                callbacks=custom_callbacks,
                devices=devices,
                accelerator="gpu",
                # strategy=ddp_strategy, # TODO enable when using multiple GPUs
                num_sanity_val_steps=0,
                accumulate_grad_batches=4,
                # precision="16-mixed",  # TODO: change to 16 for faster computation
            )
        else:
            print("No GPUs available. Training on CPU.")
            trainer = Trainer(
                max_epochs=num_epochs,  # TODO: change  self._num_epochs for testing
                logger=torch_logger,
                callbacks=custom_callbacks,
                accelerator="cpu",
                num_sanity_val_steps=0,
                accumulate_grad_batches=4,
            )

        print(f"Number of GPUs: {devices}")
        custom_dm = rep.CustomDataModule(
            training_data=training_data,
            batch_size=batch_size,
            num_workers=number_of_workers,
            cktp_path=ckpt_path,
            train_partition_prop=train_partition_prop,
            checkpoint_dir=output_path,
        )

        model = rep.GloVeLightningModel(
            num_tcr=num_tcr,
            emb_size=emb_size,
            eval_epoch=eval_epoch,
            bioid_df=training_bioids,
            jl_init=jl_init,
            l1_lambda=l1_lambda,
            train_partition_prop=train_partition_prop,
            learning_rate=learning_rate,
        )
        trainer.fit(model=model, datamodule=custom_dm, ckpt_path=ckpt_path)

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
    apps.run_plugin(logs.ConfigureApplication)
    apps.run_plugin(Application)
