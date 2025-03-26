import numpy as np
from scipy.sparse import csr_matrix

np.random.seed(0)


def create_sparse_co_occurence_matrix(df):
    # Assuming df is the co-occurrence Pandas DataFrame with columns "row_index", "column_index", and "vals"
    entries = [(row["row_index"], row["column_index"], row["vals"]) for _, row in df.iterrows()]

    # Create the sparse matrix
    print("Create the sparse matrix")
    rows, cols, data = zip(*entries, strict=False)
    print(f"Max row index: {max(rows)}")
    print(f"Max column index: {max(cols)}")
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))

    return sparse_matrix


def create_sparse_donor_tcr_matrix(df):
    # Assuming df is a Pandas DataFrame with 'name', 'bioIdentity', and 'monotonic_index' columns

    bioIdentities_df = df[["bioIdentity", "monotonic_index"]].drop_duplicates()

    # Get distinct names and assign consecutive indices starting from 0
    names_df = df[["name"]].drop_duplicates().reset_index(drop=True)
    names_df["name_index"] = names_df.index

    # Join the dataframes to get the indices for the sparse matrix
    df_joined = df.merge(names_df, on="name", how="inner")

    # Collect the shape of the sparse matrix
    num_names = len(names_df)
    num_bioIdentities = len(bioIdentities_df)
    print("Num. distinct names: ", num_names, "Num. distinct bioIds: ", num_bioIdentities)

    # Collect the entries for the sparse matrix
    # Ensure that the 'monotonic_index' is used as the column index for the matrix
    entries = [(row["name_index"], row["monotonic_index"], 1) for _, row in df_joined.iterrows()]

    # Create the sparse matrix
    print("Create the sparse matrix")
    if entries:
        rows, cols, data = zip(*entries, strict=False)
    else:
        rows, cols, data = [], [], []
    print(f"Max row index: {max(rows, default=-1)}")
    print(f"Max column index: {max(cols, default=-1)}")
    sparse_matrix = csr_matrix(
        (data, (rows, cols)), shape=(max(rows, default=-1) + 1, max(cols, default=-1) + 1)
    )

    return sparse_matrix


def jl_transform(sparse_matrix, n_components, donor_tcr=False):
    k = sparse_matrix.shape[1]
    # d_list = [100, 300, int(math.sqrt(k))]
    d = n_components

    p = 0.5 * (1 - 1 / np.sqrt(3))  # P(0) = 2/3, P(-1) = 1/6, P(1) = 1/6

    bin_mat_1 = np.random.binomial(1, p, (d, k))
    bin_mat_2 = np.random.binomial(1, p, (d, k))
    spr_bin_mat = csr_matrix(bin_mat_1 - bin_mat_2)

    left_mat = spr_bin_mat.dot(sparse_matrix.T)
    if donor_tcr:
        print("sparse matrix is Donor by TCR X")
        fin_mat = left_mat.dot(sparse_matrix) * np.sqrt(3 / d)
    else:
        print("sparse matrix is TCR by TCR C")
        fin_mat = left_mat

    raw = fin_mat.T.toarray()  # final array shape is (num_tcr, d)
    unit_norm = raw / np.linalg.norm(raw, axis=1).reshape(-1, 1)

    return unit_norm
