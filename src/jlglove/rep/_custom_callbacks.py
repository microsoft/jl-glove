import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.io import TorchCheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sklearn.manifold import TSNE

import wandb

logger = logging.getLogger(__name__)


def move_to_cpu(obj):  # type: ignore
    """Recursively move tensors in nested structures to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    else:
        return obj


class CustomTorchCheckpointIO(TorchCheckpointIO):
    _wandb_detected: bool

    def __init__(self) -> None:
        self._wandb_detected = os.environ.get("WANDB_API_KEY") is not None

    def save_checkpoint(self, checkpoint, path, storage_options=None):  # type: ignore
        # Create the directory if it doesn't exist
        try:
            logger.warning(f"Creating directory for {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)  # noqa
        except PermissionError:
            # Handle the permission error by appending a tilde to the path
            new_dir = os.path.join("~", os.path.dirname(path).lstrip(os.sep))  # noqa
            new_dir = os.path.expanduser(new_dir)  # noqa: PTH111
            print(f"Permission denied for {path}. Trying {new_dir} instead.")
            os.makedirs(new_dir, exist_ok=True)  # noqa: PTH103
            # Update the path to reflect the new directory
            path = os.path.join(new_dir, os.path.basename(path))  # noqa

        # Resolve the path to the current checkpoint
        checkpoint_path = Path(path).resolve()

        rank_zero_info(f"Attempting to save checkpoint to {checkpoint_path}")

        # Move model and optimizer states to CPU to save memory
        rank_zero_info("Moving model to CPU")
        checkpoint["state_dict"] = move_to_cpu(checkpoint["state_dict"])

        if "optimizer_states" in checkpoint:
            rank_zero_info("Moving optimizer to CPU")
            checkpoint["optimizer_states"] = move_to_cpu(checkpoint["optimizer_states"])

        # Save the checkpoint locally in the specified path
        rank_zero_info(f"Saving checkpoint to {checkpoint_path}")
        with Path(checkpoint_path).open("wb") as f:
            torch.save(checkpoint, f, pickle_protocol=4)  # Explicitly set pickle protocol to 4
        rank_zero_info(f"Checkpoint saved locally to {checkpoint_path}")

        # Log the checkpoint to W&B using artifacts
        if self._wandb_detected:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(str(checkpoint_path))
            wandb.run.log_artifact(artifact)  # type: ignore
            rank_zero_info(f"Checkpoint logged to W&B as an artifact: {checkpoint_path}")

        parent_directory = os.path.dirname(checkpoint_path)  # noqa: PTH120
        files_and_dirs = os.listdir(parent_directory)  # noqa: PTH208
        rank_zero_info(f"Contents of the checkpoint directory {parent_directory}:")
        for item in files_and_dirs:
            rank_zero_info(item)


class EpochDurationPrinter(Callback):
    _wandb_detected: bool

    def __init__(self, bioid_df):  # type: ignore
        self.bioid_df = bioid_df
        self._wandb_detected = os.environ.get("WANDB_API_KEY") is not None

    def on_train_start(self, trainer, pl_module):  # type: ignore
        global_rank = trainer.global_rank
        if global_rank == 0:
            rank_zero_info("Training is starting...")
            bioid_csv_file = ".tmp/bioids/bioid_df.csv"
            Path(".tmp/bioids/").mkdir(exist_ok=True)
            self.bioid_df.to_csv(bioid_csv_file, index=False)

            if self._wandb_detected:
                rank_zero_info("Adding bioid_df to wandb...")
                artifact = wandb.Artifact("dataset", type="dataset")
                artifact.add_file(bioid_csv_file)
                wandb.log_artifact(artifact)

        self.calculate_initial_loss(
            pl_module,
            dataloader=trainer.datamodule.train_dataloader(),  # type: ignore
            data="train",
            global_rank=global_rank,
        )
        if pl_module.train_partition_prop < 1.0:  # type: ignore
            self.calculate_initial_loss(
                pl_module,
                dataloader=trainer.datamodule.val_dataloader(),  # type: ignore
                data="validation",
                global_rank=global_rank,
            )
            self.calculate_initial_loss(
                pl_module,
                dataloader=trainer.datamodule.test_dataloader(),  # type: ignore
                data="test",
                global_rank=global_rank,
            )

    def calculate_initial_loss(self, pl_module, dataloader, data, global_rank):  # type: ignore
        # Make sure the model is in eval mode for this step
        if dataloader is None:
            rank_zero_info(f"No {data} dataloader provided. Skipping initial loss calculation.")
            return
        rank_zero_info(f"Calculating initial loss for {data}")
        pl_module.eval()
        with torch.no_grad():
            total_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                # Ensure the batch is on the correct device
                batch = [item.to(pl_module.device) for item in batch]
                loss = pl_module.compute_loss(batch, regularization=True)
                total_loss += loss.item()
                num_batches += 1
        initial_loss = total_loss / num_batches
        # Log the initial loss
        if global_rank == 0:
            rank_zero_info(f"{data} loss before training: {initial_loss}")
            if self._wandb_detected:
                wandb.log({f"initial_{data}_loss": initial_loss})

        # Make sure the model is back in train mode
        pl_module.train()

    def on_train_epoch_start(self, trainer, pl_module):  # type: ignore
        self.start_time = time.time()
        rank_zero_info(f"Starting epoch:{trainer.current_epoch}")

    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore
        duration = time.time() - self.start_time
        rank_zero_info(f"Epoch {trainer.current_epoch} duration: {duration:.2f} seconds")
        # Run the test set
        if pl_module.train_partition_prop == 1.0:
            return None
        # TODO: change the condition for the testing epochs
        if trainer.current_epoch % pl_module.eval_epoch == 0 or trainer.current_epoch <= 20:  # type: ignore
            rank_zero_info("Evaluating on test set...")
            self.log_test_loss(trainer, pl_module)

    def log_test_loss(self, trainer, pl_module):  # type: ignore
        pl_module.eval()
        with torch.no_grad():
            total_loss = 0.0
            num_batches = 0
            for batch in trainer.datamodule.test_dataloader():
                # Move batch to the correct device
                batch = [item.to(pl_module.device) for item in batch]
                loss = pl_module.test_step(batch, batch_idx=None)
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                rank_zero_info(f"Test loss after epoch {trainer.current_epoch}: {avg_loss}")
        # Switch back to training mode
        pl_module.train()

    def on_train_end(self, trainer, pl_module):  # type: ignore
        rank_zero_info("Training is done!")


class EmbeddingPlotterCallback(Callback):
    _wandb_detected: bool

    def __init__(self, bioid_df, eval_epochs):  # type: ignore
        self.bioid_df = bioid_df
        self.eval_epochs = eval_epochs
        self._wandb_detected = os.environ.get("WANDB_API_KEY") is not None

    @rank_zero_only
    def log_to_wandb(self, pl_module, plot_name, plot):  # type: ignore
        if self._wandb_detected:
            print(f"saving {plot_name} to wandb...")
            wandb.log({plot_name: wandb.Image(plot)})

    @rank_zero_only
    def log_gradients(self, pl_module):  # type: ignore
        print("Logging wandb gradients...")
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                if self._wandb_detected:
                    # Log the gradient norm
                    grad_norm = param.grad.data.norm(2)
                    wandb.log({f"gradients_norms/{name}": grad_norm.item()})

                    # Convert gradients to NumPy array and log as histogram
                    grad_np = param.grad.data.cpu().numpy()
                    wandb.log({f"gradients/{name}": wandb.Histogram(grad_np)})

    @rank_zero_only
    def save_model_checkpoint(self, trainer, pl_module, epoch_loss):  # type: ignore
        logger.info(f"Saving model for {trainer.current_epoch}")
        # these are the checkpoints to resume in case of errors
        Path(".tmp/checkpoints").mkdir(exist_ok=True)
        save_path = f".tmp/checkpoints/model_weights_epoch_{trainer.current_epoch}.pth"
        torch.save(
            {
                "epoch": trainer.current_epoch,
                "model_state_dict": pl_module.state_dict(),
                "optimizer_state_dict": pl_module.trainer.optimizers[0].state_dict(),
                "loss": epoch_loss,
            },
            save_path,
        )
        if self._wandb_detected:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)  # type: ignore

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore
        self.log_gradients(pl_module)  # TODO: enable log_gradients

        if (trainer.current_epoch) % self.eval_epochs == 0:
            epoch_loss = trainer.logged_metrics.get("train_loss", None)
            print("epoch_loss: ", epoch_loss)
            plot_labels = True

            final_embeddings = pl_module.get_embeddings()  # type: ignore
            max_tcrs = 10000  # TODO make this a hyperparameter
            if final_embeddings.shape[0] > max_tcrs:
                print(f"selecting {max_tcrs} random embeddings")
                plot_labels = False
                random_indices = np.random.permutation(final_embeddings.shape[0])[:max_tcrs]  # noqa: NPY002
                final_embeddings = final_embeddings[random_indices]

            print(f"Computing TSNE for {trainer.current_epoch}")
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(final_embeddings)
            # sub_title = f'epoch={trainer.current_epoch}-loss={epoch_loss:.2f}'
            sub_title = f"epoch={trainer.current_epoch}"
            tsne_plot = pl_module.plot_embeddings(  # type: ignore
                embeddings=tsne_results,
                title="t-SNE " + sub_title,
                bioid_df=self.bioid_df,
                plot_labels=plot_labels,
            )
            # Use the helper methods to log and save only on rank 0
            self.log_to_wandb(plot_name="t-SNE", plot=tsne_plot, pl_module=pl_module)
            Path(".tmp/plots").mkdir(exist_ok=True)
            plt.savefig(f".tmp/plots/tsne-{sub_title}.png")
            plt.clf()

            print(f"Computing UMAP for {trainer.current_epoch}")
            umap_results = umap.UMAP(random_state=42).fit_transform(final_embeddings)
            umap_plot = pl_module.plot_embeddings(  # type: ignore
                embeddings=umap_results,
                title="UMAP " + sub_title,
                bioid_df=self.bioid_df,
                plot_labels=plot_labels,
            )
            self.log_to_wandb(plot_name="UMAP", plot=umap_plot, pl_module=pl_module)
            plt.savefig(f".tmp/plots/umap-{sub_title}.png")
            plt.clf()

            self.save_model_checkpoint(trainer=trainer, pl_module=pl_module, epoch_loss=epoch_loss)
