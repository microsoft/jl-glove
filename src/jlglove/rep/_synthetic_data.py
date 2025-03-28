from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score

from ._compute_jl import (
    create_sparse_co_occurence_matrix,
    create_sparse_donor_tcr_matrix,
    jl_transform,
)


class SyntheticDataGenerator:
    def __init__(self, n: int, j: int, k: int, save_path: Path):
        self.n = n
        self.j = j
        self.k = k
        self.save_path = save_path
        # N = 5000 # num samples # J = 500 # num TCRs # K = 3# num of latent factors

    def generate(self):
        """
        ## Assume ToyData = BS.

        # B is a J * K matrix
        # S is a K * N matrix
        # K <= J (ideally low rank decomposition)
        # B: captures TCR structure
        # S: captures TCR population structure among individuls.
        """
        self.save_path.mkdir(parents=True, exist_ok=True)

        B = self.generate_tcr_structure()
        S = self.generate_tcr_pop_structure()
        ToyData = np.matmul(B, S).T
        print(ToyData.shape)

        ToyData_bin = self.binarize_data(ToyData)
        print("ToyData_bin range: ", np.min(ToyData_bin), np.max(ToyData_bin))
        print("ToyData_bin mean: ", np.mean(ToyData_bin))
        print("ToyData_bin shape: ", ToyData_bin.shape)

        non_zero_list = [(int(x), int(y)) for x, y in zip(*np.nonzero(ToyData_bin), strict=False)]
        rep_df = pd.DataFrame(non_zero_list, columns=["name", "bioIdentity"])
        training_data_url = f"{self.save_path}/Glove_Synthentic_Data_500_rep.parquet"
        print("Saving training_data to: ", training_data_url)
        rep_df.to_parquet(training_data_url, index=False)

        toy_t2, bi_df, seqs_t2 = self.compute_co_occurence(rep_df)
        basename = "Glove_Synthentic_Data_500"
        t2_url = f"{self.save_path}/{basename}_t2.parquet"
        print("Saving training_data to: ", t2_url)
        # seqs_t2 = seqs_t2.reset_index(drop=True)
        # dd.from_pandas(seqs_t2, npartitions=10).to_parquet(t2_url) # For testing training with proportion_prop
        seqs_t2.to_parquet(t2_url, index=False)

        bi_df_with_binary_columns = self.calculate_cluster_labels(B, toy_t2, bi_df)
        bi_df_url = f"{self.save_path}/{basename}_BioId_with_ES.parquet"
        print("Saving training_bioids to: ", bi_df_url)
        bi_df_with_binary_columns.to_parquet(bi_df_url, index=False)

        # Compute JL
        C_sparse = create_sparse_co_occurence_matrix(seqs_t2)
        print("co-occurrence sparse: ", C_sparse.shape)

        ## Computer Total TCR Occurrence
        tcr_total_occurrence = self.compute_total_tcr_occurrence(rep_df, bi_df)

        C_jl = jl_transform(C_sparse, n_components=100, donor_tcr=False)
        column_names = [f"JL_Col{i}" for i in range(C_jl.shape[1])]
        final_embeddings_pd = pd.DataFrame(C_jl, columns=column_names)
        final_embeddings_pd.insert(0, "monotonic_index", final_embeddings_pd.index)
        final_embeddings_pd.insert(
            len(final_embeddings_pd.columns), "Total TCR Occurrence", tcr_total_occurrence
        )

        jl_bioid = pd.merge(
            final_embeddings_pd, bi_df_with_binary_columns, how="inner", on="monotonic_index"
        )
        print("jl_bioid shape: ", jl_bioid.shape)
        jl_bioid_url = f"{self.save_path}/{basename}_BioId_with_ES_JL.parquet"
        print("Saving JL training_bioids to: ", jl_bioid_url)
        jl_bioid.to_parquet(jl_bioid_url, index=False)

        return

    def compute_total_tcr_occurrence(self, df, bio_df):
        # Assuming df and bio_df are Pandas DataFrames
        # Join the DataFrames on 'bioIdentity'
        seqs = pd.merge(df, bio_df, on="bioIdentity", how="inner")

        # Select distinct pairs of 'name' and 'monotonic_index'
        seqs_uniques = seqs[["name", "monotonic_index"]].drop_duplicates()

        # Join again on 'monotonic_index'
        seqs_uniques_filtered = pd.merge(seqs_uniques, bio_df, on="monotonic_index", how="inner")

        # Number of unique bioIdentities
        n_bioIds = seqs_uniques_filtered["bioIdentity"].nunique()

        # Assert that the number of unique bioIdentities is the same as in bio_df
        assert n_bioIds == bio_df["bioIdentity"].nunique(), (
            "Mismatch in unique bioIdentities count"
        )

        # Number of non-zero entries
        n_nonzero_entries = len(seqs_uniques_filtered)

        # Number of unique names
        n_samples = seqs_uniques_filtered["name"].nunique()
        print(f"Number of samples: {n_samples}")

        # Calculate total possible entries
        n_entries = n_samples * n_bioIds

        # Calculate sparsity
        r_sparsity = n_nonzero_entries / n_entries
        print(f"Sparsity of the donor-tcr matrix: {r_sparsity}")
        donor_tcr = create_sparse_donor_tcr_matrix(seqs_uniques_filtered)
        donor_counts_for_tcr = donor_tcr.T.dot(np.ones((donor_tcr.shape[0], 1)))
        # tcr_counts_for_donor = donor_tcr.dot(np.ones((donor_tcr.shape[1], 1)))
        return donor_counts_for_tcr

    def calculate_cluster_labels(self, B, toy_t2, bi_df):
        g = sns.clustermap(
            toy_t2, figsize=(5, 4), yticklabels=False, xticklabels=False, col_cluster=True
        )
        row_clusters = g.dendrogram_row.linkage
        clusters = fcluster(row_clusters, self.k, criterion="maxclust")
        print("clusters: ", np.unique(clusters))
        ARI = adjusted_rand_score(B.argmax(axis=1), clusters)
        print("ARI: ", ARI)
        clusters_list = clusters.tolist()

        labels_df = pd.DataFrame(
            [(int(i), int(label)) for i, label in enumerate(clusters_list)],
            columns=["monotonic_index", "label"],
        )
        bi_df_label = bi_df.merge(labels_df, on="monotonic_index", how="inner")
        # Create binary columns for CMV, Parvo, and Covid
        for virus, label in [("CMV", 1), ("Parvo", 2), ("Covid", 3)]:
            bi_df_label[virus] = (bi_df_label["label"] == label).astype(int)
            print(virus, bi_df_label[virus].sum())

        return bi_df_label

    def binarize_data(self, ToyData):
        def binary(t):
            return np.random.binomial(n=1, p=t, size=1)[0]

        vfunc = np.vectorize(binary)
        ToyData_bin = vfunc(ToyData)
        return ToyData_bin

    def generate_tcr_pop_structure(self):
        S = np.zeros((self.k, self.n))
        ## Create S in K * N
        S_expos = int(self.n / 2)
        for i in range(S_expos):
            alpha = [1] * self.k  ## change this to create sparsity
            q_K = np.random.dirichlet(alpha, size=1)[0]
            # print(q_K)
            S[0][i] = q_K[0]
            S[1][i] = q_K[1]
            S[2][i] = q_K[2]
        return S

    def generate_tcr_structure(self):
        B = np.empty((self.j, self.k))
        ## Create T in J * K

        for j in range(self.j):
            alpha = [0.01] * self.k
            # alpha = 0.01, 0.1, 0.5, 1 varies level of sparsity, smaller => sparse
            q_K = np.random.dirichlet(alpha, size=1)[0]
            B[j][0] = q_K[0]
            B[j][1] = q_K[1]
            B[j][2] = q_K[2]
        return B

    def compute_co_occurence(self, df):
        # Get unique bioIdentity values and sort them
        biods = df["bioIdentity"].unique()
        biods.sort()

        # Create a DataFrame with bioIdentity and corresponding index
        bi_df = pd.DataFrame({"bioIdentity": biods, "index": range(len(biods))})
        # Join the original DataFrame with the bioIdentity-index DataFrame
        seqs_ind = df.merge(bi_df, on="bioIdentity").drop("bioIdentity", axis=1)

        # Create two versions of the DataFrame with renamed columns for row and column indices
        seqs_row = seqs_ind.rename(columns={"index": "row_index"})
        seqs_col = seqs_ind.rename(columns={"index": "column_index"})

        # Perform a self-join on the 'name' column, filtering out self-joins
        seqs_join = pd.merge(seqs_row, seqs_col, on="name")
        seqs_join = seqs_join[seqs_join["row_index"] != seqs_join["column_index"]]

        # Group by row_index and column_index and count occurrences
        seqs_join = (
            seqs_join.groupby(["row_index", "column_index"]).size().reset_index(name="vals")
        )
        # Create a COO matrix from the counts
        t2 = coo_matrix(
            (seqs_join["vals"], (seqs_join["row_index"], seqs_join["column_index"])),
            shape=(len(biods), len(biods)),
        )
        data = t2.toarray()
        # Assert the matrix is symmetric
        assert np.array_equal(data, data.T)
        # Rename the 'index' column to 'monotonic_index'
        bi_df = bi_df.rename(columns={"index": "monotonic_index"})

        return data, bi_df, seqs_join
