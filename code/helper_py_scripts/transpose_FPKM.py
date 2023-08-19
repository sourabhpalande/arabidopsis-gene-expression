import pandas as pd
from tqdm import tqdm
from time import perf_counter

def transpose_FPKM(infile="../../data/gene_FPKM_200501.csv",
                    outfile="../../data/gene_FPKM_transposed",
                    ext=".gzip"):
    '''Transpose the original gene_FPKM file
        - Original file has genes as rows, samples as columns. For ML, it is more convenient to have samples as rows, genes as columns. For a large file, transposing on-the-fly can be time consuming, so we want to transpose and store the stored dataframe as a csv or parquet file.
    
        Parameters
        ----------
        infile : str, default : "../data/gene_FPKM_200501.csv"
            path to the original gene expression file.

        outfile : str, default : "../data/gene_FPKM_transposed"
            Output filename (without extension)

        ext : str, default : ".gzip"
            Extension is used to determine the filetype of the output file
            can be either ".csv" or ".gzip"

        Returns
        -------

        None

    '''
    df_list = []
    df_array = pd.DataFrame()

    tic = perf_counter()

    for df in tqdm(pd.read_csv(infile, low_memory=False, chunksize = 10000)):
        print(' --- Complete')
        df_list.append(df)

    df_array = pd.concat(df_list)
    print(df_array.shape)
    print(f"Columns before transpose: {df_array.columns}")

    df_transposed = df_array.set_index("Sample").transpose()
    df_transposed = df_transposed.rename_axis("SampleID")\
        .rename_axis(None, axis="columns").reset_index()

    print(df_transposed.shape)
    print(f"Column names: {df_transposed.columns}")
    print(df_transposed["SampleID"])
    print(df_transposed.head(2))

    if ext == ".csv":
        df_transposed.to_csv(outfile+".csv", index=False)
        print("Wrote transposed dataframe to csv")
    elif ext == ".gzip":
        df_transposed.to_parquet(outfile+".gzip",
                                 compression="gzip", index=None)
        print("Wrote transposed dataframe to parquet")
    else:
        print("extension not recognized. Options: '.csv' and '.gzip'")

    print(f"time elapsed: {perf_counter() - tic}")

    return None


if __name__ == "__main__":
    print("Transposing gene FPKM file...")
    transpose_FPKM()
    print("Done...")
