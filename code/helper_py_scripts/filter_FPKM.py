from time import perf_counter
from tqdm import tqdm
import pandas as pd



def filter_fpkm(infile="../../data/gene_FPKM_transposed.gzip",
                outfile="../../data/gene_FPKM_transposed_UMR75.gzip",
                metafile="../../data/metadata_UMR75.csv"):
    '''Filter RNAseq samples using SampleID from metadata file
        - metadata file contains SampleIDs filtered by unique mapped rate. We only want RNAseq profiles corresponding to those samples for later.
    
        Parameters
        ----------
        infile : str, default : "../data/gene_FPKM_transposed.gzip"
            path to the RNAseq file.

        outfile : str, default : "../data/gene_FPKM_transposed_UMR75.gzip"
            Filename for the filtered RNAseq data

        metafile : str, default : "../data/metadata_UMR75.csv"
            Filename for the filtered metadata file

        Returns
        -------

        None

    '''
    tic = perf_counter()
    mdf = pd.read_csv(metafile)
    print(f"metadata shape: {mdf.shape}")
    sample_ids = mdf["SampleID"].tolist()
    rnadf = pd.read_parquet(infile)
    print(f"RNAseq data shape before filtering: {rnadf.shape}")
    rnadf.query('SampleID in @sample_ids', inplace=True)
    print(f"RNAseq data shape after filtering: {rnadf.shape}")
    rnadf = pd.merge(mdf, rnadf, on='SampleID')
    print("Final DataFrame Shape: {rnadf.shape}")
    rnadf.to_parquet(outfile, compression="gzip", index=None)
    print("Wrote filtered RNAseq data to file")
    print(f"Time elapsed: {perf_counter() - tic}")



def filter_and_transpose_fpkm(infile="../../data/gene_FPKM_200501.csv",
                outfile="../../data/gene_FPKM_transposed_UMR75.gzip",
                metafile="../../data/metadata_UMR75.csv"):
    '''Filter RNAseq samples using SampleID from metadata file
        - metadata file contains SampleIDs filtered by unique mapped rate. We only want RNAseq profiles corresponding to those samples for later.
    
        Parameters
        ----------
        infile : str, default : "../data/gene_FPKM_200501.csv"
            path to the RNAseq file.

        outfile : str, default : "../data/gene_FPKM_transposed_UMR75.gzip"
            Filename for the filtered RNAseq data

        metafile : str, default : "../data/metadata_UMR75.csv"
            Filename for the filtered metadata file

        Returns
        -------

        None

    '''
    tic = perf_counter()
    ext = outfile.split(".")[-1]
    print(f"outfile extension: {ext}")

    mdf = pd.read_csv(metafile)
    print(f"metadata shape: {mdf.shape}")
    print(f"metadata columns: {mdf.columns}")
    sample_ids = (mdf["SampleID"].tolist()).insert(0, "Sample")

    df_list = []
    df_array = pd.DataFrame()
    for df in tqdm(pd.read_csv(infile, low_memory=False,
                               usecols=sample_ids, chunksize = 10000)):
        print(' --- Complete')
        df_list.append(df)

    df_array = pd.concat(df_list)
    df_transposed = df_array.set_index("Sample").transpose()
    df_transposed = df_transposed.rename_axis("SampleID")\
        .rename_axis(None, axis="columns").reset_index()

    print(f"Shape before transpose: {df_array.shape}")
    print(f"Shape after transpose: {df_transposed.shape}")
    print(f"Column names: {df_transposed.columns}")
    print(df_transposed["SampleID"])
    print(df_transposed.head(2))

    # Merge filtered and transposed FPKM data with meta data
    df_transposed = pd.merge(mdf, df_transposed, on='SampleID')
    print(f"Final DataFrame Shape: {df_transposed.shape}")

    if ext == "csv":
        df_transposed.to_csv(outfile+".csv", index=False)
        print("Wrote transposed dataframe to csv")
    elif ext == "gzip":
        df_transposed.to_parquet(outfile, compression="gzip", index=None)
        print("Wrote transposed dataframe to parquet")
    else:
        print("extension not recognized. Options: '.csv' and '.gzip'")

    print(f"time elapsed: {perf_counter() - tic}")



if __name__ == "__main__":
    print("Filtering RNAseq samples...")
    filter_and_transpose_fpkm()
    print("Done...")
