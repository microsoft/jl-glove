import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class GloVeLightningModel(LightningModule):
    def __init__(
        self,
        num_tcr,  # type: ignore
        emb_size,  # type: ignore
        bioid_df,  # type: ignore
        jl_init=False,  # type: ignore
        train_partition_prop=1.0,  # type: ignore
        l1_lambda=0,  # type: ignore
        x_max=100,  # type: ignore
        alpha=0.75,  # type: ignore
        learning_rate=0.05,  # type: ignore
        batch_size=1,  # type: ignore
        num_workers=0,  # type: ignore
        eval_epoch=100,  # type: ignore
    ):
        super().__init__()
        self.wi = nn.Embedding(num_tcr, emb_size)
        self.wj = nn.Embedding(num_tcr, emb_size)
        self.bi = nn.Embedding(num_tcr, 1)
        self.bj = nn.Embedding(num_tcr, 1)

        if jl_init:
            # Convert specific columns from the DataFrame to a tensor
            print("Using JL embeddings for initialization.")
            column_names = [f"JL_Col{i}" for i in range(emb_size)]
            sorted_bioid_df = bioid_df.sort_values(by="monotonic_index")
            jl_embeddings = sorted_bioid_df[column_names].to_numpy()
            wi_init = torch.tensor(jl_embeddings, dtype=torch.float32)
            wj_init = torch.tensor(jl_embeddings, dtype=torch.float32)

            # Make sure the tensor is of the correct shape (output_features x input_features)
            wi_init = wi_init.view(num_tcr, emb_size)
            wj_init = wj_init.view(num_tcr, emb_size)

            # Initialize weights with the tensor
            self.wi.weight = nn.Parameter(wi_init)
            self.wj.weight = nn.Parameter(wj_init)

            tcr_occurrence = sorted_bioid_df["Total TCR Occurrence"].to_numpy()
            tcr_occurrence_normalized = tcr_occurrence / np.linalg.norm(tcr_occurrence)

            bi_init = torch.tensor(tcr_occurrence_normalized, dtype=torch.float32)
            bj_init = torch.tensor(tcr_occurrence_normalized, dtype=torch.float32)

            bi_init = bi_init.view(num_tcr, 1)
            bj_init = bj_init.view(num_tcr, 1)

            self.bi.weight = nn.Parameter(bi_init)
            self.bj.weight = nn.Parameter(bj_init)

        else:
            print("Using random for initialization.")
            self.wi.weight.data.uniform_(
                -1, 1
            )  # TODO: Should the weights be constrained to positive values?
            self.wj.weight.data.uniform_(-1, 1)

            self.bi.weight.data.zero_()
            self.bj.weight.data.zero_()

        self.b0 = 0.1
        self.b = 1 / 4
        self.alpha = 1 / 2

        self.l1_lambda = l1_lambda  # L1-regularization coefficient to be added to GloVe loss
        self.train_partition_prop = train_partition_prop
        self.eval_epoch = eval_epoch

        # Store hyperparameters
        self.save_hyperparameters()

    def weighting_func(self, x):  # type: ignore
        return self.b0 + self.b * (x**self.alpha)

    def forward(self, i_indices, j_indices):  # type: ignore
        w_i = self.wi(i_indices).squeeze()
        w_j = self.wj(j_indices).squeeze()
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

    def compute_loss(self, batch, regularization=True):  # type: ignore
        i_indices, j_indices, counts = batch
        x = self.forward(i_indices, j_indices)
        y = torch.log(counts)
        loss = torch.pow(x - y, 2)
        loss = torch.mul(self.weighting_func(counts), loss).mean()

        if regularization:
            w_i = self.wi(i_indices).squeeze()
            w_j = self.wj(j_indices).squeeze()
            loss += self.l1_lambda * (torch.norm(w_i, p=1) + torch.norm(w_j, p=1))

        return loss

    def training_step(self, batch, batch_idx):  # type: ignore
        loss = self.compute_loss(batch, regularization=True)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        if self.train_partition_prop == 1.0:
            return None
        # TODO: change the condition for the validation epochs
        # if self.current_epoch % self.eval_epoch == 0 or self.current_epoch <= 20:
        if self.current_epoch % self.eval_epoch == 0 or self.current_epoch <= 10000:
            loss = self.compute_loss(
                batch, regularization=False
            )  # # Only calculate the main loss (no L1 regularization)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=False,
            )
            return loss

    def test_step(self, batch, batch_idx):  # type: ignore
        loss = self.compute_loss(
            batch, regularization=False
        )  # # Only calculate the main loss (no L1 regularization)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=False,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
        )  # Used in the GloVe algorithm works better that Adam for Sparse data
        # TODO: check other optimizers
        # Used in the GloVe algorithm works better that Adam for Sparse data
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    # Add your validation_step, test_step, etc. as needed

    def get_embeddings(self) -> torch.Tensor:
        # get embeddings
        embeddings_i = self.wi.weight.data.cpu().numpy()
        embeddings_j = self.wj.weight.data.cpu().numpy()

        # Average the two embeddings
        final_embeddings = (embeddings_i + embeddings_j) / 2.0
        return final_embeddings  # type: ignore

    def get_idx(self, label, bioid_df):  # type: ignore
        return bioid_df[bioid_df[label] == 1]["monotonic_index"].values

    def plot_embeddings(self, bioid_df, embeddings, title, plot_labels=False) -> plt:  # type: ignore
        plt.figure(figsize=(10, 10))
        labels = ["CMV", "Parvo", "Covid"]
        colors = sns.color_palette("husl", len(labels) + 1)
        plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            edgecolors=colors[0],
            cmap="prism",
            c="w",
            marker="o",
            label=f"All ({len(embeddings)})",
        )

        if plot_labels:
            for c_idx, lb in enumerate(labels):
                idx = self.get_idx(lb, bioid_df)
                plt.scatter(
                    embeddings[idx, 0],
                    embeddings[idx, 1],
                    edgecolors=colors[c_idx + 1],
                    cmap="prism",
                    c="w",
                    marker="o",
                    label=f"{lb} ({len(idx)})",
                )
        plt.legend(loc=(1, 0.2))
        plt.title(title)
        plt.tight_layout()
        return plt

    @staticmethod
    def combine_pooling(embeddings, stat_list: tuple[str, ...] = ("max", "mean")):  # type: ignore
        embeddings_stacked = np.vstack(embeddings)
        stat_arr = list()

        for stat in stat_list:
            if stat == "min":
                stat_arr.append(np.min(embeddings_stacked, axis=0))
            if stat == "max":
                stat_arr.append(np.max(embeddings_stacked, axis=0))
            if stat == "mean":
                stat_arr.append(np.mean(embeddings_stacked, axis=0))
            if stat == "2nd_moment":
                stat_arr.append(np.mean(np.power(embeddings_stacked, 2), axis=0))
            if stat == "3rd_moment":
                stat_arr.append(np.mean(np.power(embeddings_stacked, 3), axis=0))
            if stat == "1st_quantile":
                stat_arr.append(np.quantile(embeddings_stacked, q=0.25, axis=0))
            if stat == "median":
                stat_arr.append(np.quantile(embeddings_stacked, q=0.5, axis=0))
            if stat == "3rd_quantile":
                stat_arr.append(np.quantile(embeddings_stacked, q=0.75, axis=0))
            if stat == "sum":
                stat_arr.append(np.sum(embeddings_stacked, axis=0))
            if stat == "abs_sum":
                stat_arr.append(np.sum(embeddings_stacked, axis=0))

        return np.concatenate(stat_arr)

    @staticmethod
    def check_bioids(df: pd.DataFrame, bioid_df: pd.DataFrame) -> bool:
        if "monotonic_index" not in df.columns:
            print("df missing monotonic index column")
            return False

        # Select distinct bioIdentity from both DataFrames
        distinct_df_bioids = df["bioIdentity"].unique()
        distinct_bioid_df_bioids = bioid_df["bioIdentity"].unique()

        # Check if all bioIdentity values in df are in bioid_df
        is_subset = pd.Series(distinct_df_bioids).isin(distinct_bioid_df_bioids).all()

        # Print the result
        if is_subset:
            print("All bioIdentity values in df are a subset of those in bioid_df.")
        else:
            print("There are bioIdentity values in df that are not in bioid_df.")

        return is_subset

    @staticmethod
    def merge_aggregate_embeddings(  # type: ignore
        df,  # type: ignore
        bioid_df,  # type: ignore
        embeddings,  # type: ignore
        label=None,  # type: ignore
        es_idx=None,  # type: ignore
        sub_label=None,  # type: ignore
        aggregate_fun=combine_pooling,  # type: ignore
        aggregate_stat_list=("min", "max"),  # type: ignore
        names_to_select=None,  # type: ignore
    ):
        """Merge and aggregate embeddings."""
        print(
            "label: ",
            label,
            "sub_label: ",
            sub_label,
            "aggregate_fun: ",
            aggregate_fun,
            "aggregate_stat_list: ",
            aggregate_stat_list,
        )

        # Merge dataframes on 'bioIdentity'
        # and create 'embedding' column by mapping 'idx' to embeddings
        if not GloVeLightningModel.check_bioids(df=df, bioid_df=bioid_df):
            print("Merging dataframes on 'bioIdentity' and creating 'embedding' column")
            if "monotonic_index" in df.columns:
                df = df.drop("monotonic_index", axis=1)
            df_idx = pd.merge(df, bioid_df, on="bioIdentity", how="inner").rename(
                columns={"monotonic_index": "idx"}
            )
        else:
            print("Skipping merge")
            df_idx = df.rename(columns={"monotonic_index": "idx"})

        df_idx["embedding"] = df_idx["idx"].map(embeddings)

        # filter the rows if names_to_select is provided
        if names_to_select is not None:
            print("Filtering rows based on names_to_select")
            df_idx = df_idx[df_idx["name"].isin(names_to_select)]

        # Prepare for groupby operation
        if es_idx is not None:
            df_idx["es_count"] = df_idx["idx"].isin(es_idx).astype(int)  # type: ignore

        # Aggregate embeddings by 'name' with a single groupby operation
        aggregation_functions = {
            "embedding": lambda x: aggregate_fun(x, stat_list=aggregate_stat_list),
            "totalUniqueArrangments": "first",
            "templates": "sum",
        }

        if es_idx is not None:
            aggregation_functions["es_count"] = "sum"

        if label:
            aggregation_functions[label] = "first"

        if sub_label:
            aggregation_functions[sub_label] = "first"

        # Perform the groupby and aggregation in one step
        print("Performing groupby and aggregation in one step")
        df_final = (
            df_idx.groupby("name")
            .agg(aggregation_functions)
            .reset_index()
            .rename(columns={"templates": "totalTemplates"})
        )

        return df_final
