import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import perf_counter


def filter_genes_by_orthos(rnafile="../data/gene_FPKM_transposed_UMR75.gzip",
            orthofile="../data/Angiosperm_RNAseq_clean.csv"):
    '''Filter RNAseq columns by orthogroups from the Angiosperm data
        - 
    
        Parameters
        ----------
        rnafile : str, default : "../data/gene_FPKM_transposed_UMR75.gzip"
            path to the filtered RNAseq data file.

        orthofile : str, default : "../data/Angiosperm_RNAseq_clean.csv"
            Filename for the file containing genes corresponding to orthogroups

        Returns
        -------

        None

    '''
    tic = perf_counter()
    angiodf = pd.read_csv(orthofile)
    print(f"angiosperm data shape: {angiodf.shape}")
    gene_names = (angiodf.columns[1:]).tolist()
    arabidf = pd.read_parquet(rnafile)
    print(f"RNAseq data shape: {arabidf.shape}")
    toc = perf_counter()
    print(f"Data loaded. Time elapsed: {toc - tic}")
    colnames = (arabidf.columns[:14]).tolist() + gene_names
    new_arabidf = arabidf[colnames]

    print(f"New arabidopsis RNAseq data shape: {new_arabidf.shape}")
    print(f"Column filtering done. Time elapsed: {perf_counter() - tic}")

    return None



if __name__ == "__main__":
    print("Filtering RNAseq columns...")
    filter_genes_by_orthos()
    print("Done...")