import logging
import os
import random
import shutil

import dask.dataframe as dd
import pytorch_lightning as pl
import torch

# import wandb
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_data: dd.DataFrame,
        batch_size: int,
        num_workers: int,
        cktp_path: str,
        checkpoint_dir: str,
        train_partition_prop: float = 1.0,
    ):
        super().__init__()
        self.training_data = training_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ckpt_path = cktp_path
        self.local_training_data = None
        self.train_partition_prop = train_partition_prop
        self.checkpoint_dir = checkpoint_dir + "/artifacts"
        self._val_dataloader = None  # Caching validation dataloader
        self._test_dataloader = None  # Caching test dataloader

    def prepare_data(self):
        if self.ckpt_path is not None and self.trainer.is_global_zero:
            # Initialize WandB
            logger.warning("SKIPPING WANDB THINGS HERE")
            # wandb.init(project="glove_lightning")
            print("Downloading checkpoint from wandb...")
            # artifact = wandb.use_artifact(self.ckpt_path, type="model")
            # artifact_dir = artifact.download(root=self.checkpoint_dir)
            # print("Saved Checkpoint files at: ", artifact_dir)
            # files = os.listdir(artifact_dir)
            print("Files in the checkpoint directory:")
            # for file in files:
            #     print(file)
            # self.local_ckpt_path = os.path.join(artifact_dir, files[-1])
            print("Final checkpoint path: ", self.local_ckpt_path)
            # Load the checkpoint file
            checkpoint = torch.load(self.local_ckpt_path)
            print("Checkpoint Keys", checkpoint.keys())

    def clear_directory(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    def setup(self, stage=None):
        #  This is called on every GPU
        # Split the dataset into train, validation, and test here if needed
        if stage == "fit" or stage is None:
            self.local_training_data = self.training_data

    def train_dataloader(self):
        dataset = CoOccurrenceDataset(
            self.local_training_data, partition_prop=self.train_partition_prop
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,  # should be 0 if using Dask
            shuffle=True,
            collate_fn=CoOccurrenceDataset.custom_collate_fn,
            # persistent_workers=True if self.num_workers > 0 else False,
            # pin_memory=True #Did not see improvement in performance
        )
        return dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            val_partition_prop = self.train_partition_prop
            print(f"Creating validation dataloader with partion prop {val_partition_prop}")
            dataset = CoOccurrenceDataset(
                self.local_training_data, partition_prop=val_partition_prop
            )
            self._val_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,  # should be 0 if using Dask
                shuffle=False,
                collate_fn=CoOccurrenceDataset.custom_collate_fn,
                # persistent_workers=True if self._number_of_workers > 0 else False,
                # pin_memory=True Did not see improvement in performance
            )
            return self._val_dataloader
        else:
            return self._val_dataloader

    def test_dataloader(self):
        if self._test_dataloader is None:
            test_partition_prop = 1.0
            print(f"Creating test dataloader with partion prop {test_partition_prop}")
            dataset = CoOccurrenceDataset(
                self.local_training_data, partition_prop=test_partition_prop
            )
            self._test_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,  # should be 0 if using Dask
                shuffle=False,
                collate_fn=CoOccurrenceDataset.custom_collate_fn,
                # persistent_workers=True if self._number_of_workers > 0 else False,
                # pin_memory=True Did not see improvement in performance
            )
            return self._test_dataloader
        else:
            return self._test_dataloader


class CoOccurrenceDataset(Dataset):  # type: ignore
    def __init__(self, dask_df, partition_prop=1.0) -> None:
        self.dask_df = dask_df
        self.total_indices = list(range(self.dask_df.npartitions))

        # Randomly select a proportion of indices
        self.selected_indices = random.sample(
            self.total_indices, int(len(self.total_indices) * partition_prop)
        )

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, idx) -> tuple:  # type: ignore
        # Map the input index to the selected index
        actual_idx = self.selected_indices[idx]

        # Read the DataFrame corresponding to the given index
        partition_df = self.dask_df.get_partition(actual_idx).compute()

        # Extract column values
        row_indices = partition_df["row_index"].values
        column_indices = partition_df["column_index"].values
        values = partition_df["vals"].values

        # Convert to tensors
        # TODO: check if I can get away with smaller data types
        column_indices_tensor = torch.tensor(column_indices, dtype=torch.int32)
        row_indices_tensor = torch.tensor(row_indices, dtype=torch.int32)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        return (row_indices_tensor, column_indices_tensor, values_tensor)

    def custom_collate_fn(batch) -> tuple:  # type: ignore
        row_indices_list = []
        column_indices_list = []
        values_list = []

        # Iterate over the batch
        for row_indices_tensor, column_indices_tensor, values_tensor in batch:
            row_indices_list.append(row_indices_tensor)
            column_indices_list.append(column_indices_tensor)
            values_list.append(values_tensor)

        # Use torch.cat to concatenate the lists of tensors along the first dimension
        row_indices = torch.cat(row_indices_list)
        column_indices = torch.cat(column_indices_list)
        values = torch.cat(values_list)

        # Return the concatenated tensors as a tuple
        return row_indices, column_indices, values
