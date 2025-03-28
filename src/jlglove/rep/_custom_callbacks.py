import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.io import TorchCheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sklearn.manifold import TSNE

import logging

logger = logging.getLogger(__name__)


def move_to_cpu(obj):
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
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # Create the directory if it doesn't exist
        try:
            print(f"Creating directory for {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except PermissionError:
            # Handle the permission error by appending a tilde to the path
            new_dir = os.path.join("~", os.path.dirname(path).lstrip(os.sep))
            new_dir = os.path.expanduser(new_dir)  # Expand the tilde to the user's home directory
            print(f"Permission denied for {path}. Trying {new_dir} instead.")
            os.makedirs(new_dir, exist_ok=True)
            # Update the path to reflect the new directory
            path = os.path.join(new_dir, os.path.basename(path))

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
        with open(checkpoint_path, "wb") as f:
            torch.save(checkpoint, f, pickle_protocol=4)  # Explicitly set pickle protocol to 4
        rank_zero_info(f"Checkpoint saved locally to {checkpoint_path}")

        logger.warning("SKIPPING WANDB THINGS HERE")
        # Log the checkpoint to W&B using artifacts
        # artifact = wandb.Artifact("model", type="model")
        # artifact.add_file(checkpoint_path)
        # wandb.run.log_artifact(artifact)
        rank_zero_info(f"Checkpoint logged to W&B as an artifact: {checkpoint_path}")

        parent_directory = os.path.dirname(checkpoint_path)
        files_and_dirs = os.listdir(parent_directory)
        rank_zero_info(f"Contents of the checkpoint directory {parent_directory}:")
        for item in files_and_dirs:
            rank_zero_info(item)


class EpochDurationPrinter(Callback):
    def __init__(self, bioid_df):
        self.bioid_df = bioid_df

    def on_train_start(self, trainer, pl_module):
        global_rank = trainer.global_rank
        if global_rank == 0:
            rank_zero_info("Training is starting...")
            logger.warning("SKIPPING WANDB THINGS HERE")
            # bioid_csv_file = "bioid_df.csv"
            # self.bioid_df.to_csv(bioid_csv_file, index=False)
            # rank_zero_info("Adding bioid_df to wandb...")
            # artifact = wandb.Artifact("dataset", type="dataset")
            # artifact.add_file(bioid_csv_file)
            # wandb.log_artifact(artifact)

        self.calculate_initial_loss(
            pl_module,
            dataloader=trainer.datamodule.train_dataloader(),
            data="train",
            global_rank=global_rank,
        )
        if pl_module.train_partition_prop < 1.0:
            self.calculate_initial_loss(
                pl_module,
                dataloader=trainer.datamodule.val_dataloader(),
                data="validation",
                global_rank=global_rank,
            )
            self.calculate_initial_loss(
                pl_module,
                dataloader=trainer.datamodule.test_dataloader(),
                data="test",
                global_rank=global_rank,
            )

    def calculate_initial_loss(self, pl_module, dataloader, data, global_rank):
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
            logger.warning("SKIPPING WANDB THINGS HERE")
            # wandb.log({f"initial_{data}_loss": initial_loss})

        # Make sure the model is back in train mode
        pl_module.train()

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        rank_zero_info(f"Starting epoch:{trainer.current_epoch}")

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.start_time
        rank_zero_info(f"Epoch {trainer.current_epoch} duration: {duration:.2f} seconds")
        # Run the test set
        if pl_module.train_partition_prop == 1.0:
            return None
        # TODO: change the condition for the testing epochs
        if trainer.current_epoch % pl_module.eval_epoch == 0 or trainer.current_epoch <= 20:
            rank_zero_info("Evaluating on test set...")
            self.log_test_loss(trainer, pl_module)

    def log_test_loss(self, trainer, pl_module):
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

    def on_train_end(self, trainer, pl_module):
        rank_zero_info("Training is done!")


class EmbeddingPlotterCallback(Callback):
    def __init__(self, bioid_df, eval_epochs):
        self.bioid_df = bioid_df
        self.eval_epochs = eval_epochs

    @rank_zero_only
    def log_to_wandb(self, pl_module, plot_name, plot):
        print(f"saving {plot_name} to wandb...")
        logger.warning("SKIPPING WANDB THINGS HERE")
        plot.show()
        # wandb.log({plot_name: wandb.Image(plot)})

    @rank_zero_only
    def log_gradients(self, pl_module):
        print("Logging wandb gradients...")
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Log the gradient norm
                grad_norm = param.grad.data.norm(2)
                # wandb.log({f"gradients_norms/{name}": grad_norm.item()})

                # Convert gradients to NumPy array and log as histogram
                grad_np = param.grad.data.cpu().numpy()
                # wandb.log({f"gradients/{name}": wandb.Histogram(grad_np)})

    @rank_zero_only
    def save_model_checkpoint(self, trainer, pl_module, epoch_loss):
        print(f"Saving model for {trainer.current_epoch}")
        logger.warning("SKIPPING WANDB THINGS HERE")
        # TODO: make this log to local files
        # these are the checkpoints to resume in case of errors

        # save_path = f"model_weights_{wandb.run.id}_epoch_{trainer.current_epoch}.pth"
        # torch.save(
        #     {
        #         "epoch": trainer.current_epoch,
        #         "model_state_dict": pl_module.state_dict(),
        #         "optimizer_state_dict": pl_module.trainer.optimizers[0].state_dict(),
        #         "loss": epoch_loss,
        #     },
        #     save_path,
        # )
        # artifact = wandb.Artifact("model", type="model")
        # artifact.add_file(save_path)
        # wandb.run.log_artifact(artifact)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.log_gradients(pl_module)  # TODO: enable log_gradients

        if (trainer.current_epoch) % self.eval_epochs == 0:
            epoch_loss = trainer.logged_metrics.get("train_loss", None)
            print("epoch_loss: ", epoch_loss)
            plot_labels = True

            final_embeddings = pl_module.get_embeddings()
            max_tcrs = 10000  # TODO make this a hyperparameter
            if final_embeddings.shape[0] > max_tcrs:
                print(f"selecting {max_tcrs} random embeddings")
                plot_labels = False
                random_indices = np.random.permutation(final_embeddings.shape[0])[:max_tcrs]
                final_embeddings = final_embeddings[random_indices]

            print(f"Computing TSNE for {trainer.current_epoch}")
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(final_embeddings)
            # sub_title = f'epoch={trainer.current_epoch}-loss={epoch_loss:.2f}'
            sub_title = f"epoch={trainer.current_epoch}"
            tsne_plot = pl_module.plot_embeddings(
                embeddings=tsne_results,
                title="t-SNE " + sub_title,
                bioid_df=self.bioid_df,
                plot_labels=plot_labels,
            )
            # Use the helper methods to log and save only on rank 0
            self.log_to_wandb(plot_name="t-SNE", plot=tsne_plot, pl_module=pl_module)
            plt.clf()

            # print(f"Computing UMAP for {trainer.current_epoch}")
            # umap_results = umap.UMAP(random_state=42).fit_transform(final_embeddings)
            # umap_plot = pl_module.plot_embeddings(
            #     embeddings=umap_results,
            #     title="UMAP " + sub_title,
            #     bioid_df=self.bioid_df,
            #     plot_labels=plot_labels
            # )
            # self.log_to_wandb(plot_name ="UMAP",plot= umap_plot, pl_module=pl_module)
            # plt.clf()

            # self.save_model_checkpoint(trainer=trainer, pl_module=pl_module, epoch_loss=epoch_loss)
