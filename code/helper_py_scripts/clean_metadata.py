import pandas as pd


def clean_metadata(metafile="../../data/Arabidopsis_metadata.tsv",
                    ttfile="../../data/all_tissue_type.csv",
                    filtered_metafile="../../data/metadata_UMR75.csv",
                    tissue_mapfile="../../data/tissue_type_map_UMR75.csv",
                    thresh=0.75):
    '''Clean and filter metadata files
        - Filters samples from the metadata file (Arabidopsos_metadata.tsv) by applying the specified threshold to the "UniqueMappedRate" column.
        
        - Creates a list of unique "Tissue" types from the filtered samples and uses the master tissue type file (all_tissue_types.csv) to assign "VegetativeRepro" and "AboveBelow" labels to each tissue type. This mapping is written out to a file (tissue_type_map_UMR<thresh>.csv).

        - Finally, Each sample in the filtered metadata is assigned the "VegetativeRepro" and "AboveBelow" labels using the tissue type mapping. The resulting metadata is written out to a file (metadata_UMR<thresh>.csv).


        Parameters
        ----------
        metafile : str, default : "../data/Arabidopsis_metadata.tsv"
            path to the metadata csv file.

        ttfile : str, default : "../data/all_tissue_type.csv"
            Master tissue type file (contains all unique tissue types from the metadata along with the corresponding "VegetativeRepro" and "AboveBelow" labels).

        filtered_metafile : str, default : "../data/metadata_UMR75.csv"
            path to output metadata file.

        tissue_mapfile : str, default : "../data/tissue_type_map_UMR75.csv"
            path to output tissue type mapping file corresponding to the filtered metadata file.

        thresh : float, default : 0.75
            Threshold to applied to the UniqueMappedRate for filtering metadata.

        Returns
        -------

        None

    '''
    # Read metadata file
    mdf = pd.read_csv(metafile, sep="\t")
    mdf.replace("/", "Other", inplace=True)
    temp = mdf["UniqueMappedRate"].str.rstrip("%").astype(float)/100.
    mdf["UniqueMappedRate"] = temp

    print("Metadata sample: ")
    print(mdf.head(3))

    # Filter by UniqueMappedRate and write to file
    mdf_filtered = mdf[mdf["UniqueMappedRate"] >= thresh]
    print(f"shape before filtering: {mdf.shape}\n")
    print(f"shape after filtering: {mdf_filtered.shape}")

    # Create file for Unique tissue labels
    res = mdf["Tissue"].value_counts()
    tissues = res.keys()
    counts = res.ravel()
    tdf = pd.DataFrame(list(zip(tissues, counts)), columns=["Tissue", "Count"])
    print("Number of unique tisue labels: {}".format(tdf.shape[0]))

    # Add VegetativeRepro, AboveBelow labels and write to file
    tdf_old = pd.read_csv(ttfile, skipinitialspace=True).fillna("Other")
    tdf_old.replace("/", "Other", inplace=True)

    tdf_new = tdf.merge(tdf_old, on="Tissue")
    tdf_new = tdf_new.drop(["Counts", "Debatable"], axis=1)
    tdf_new.rename(columns={"Tissue.1":"TissueClean"})

    print(f"tdf columns: {tdf.columns}")
    print(f"tdf_old columns: {tdf_old.columns}")
    print(f"tdf_new columns {tdf_new.columns}")

    # tdf_new = tdf_new.rename(columns={"Tissue_Corrected":"Tissue"})
    tdf_new.to_csv(tissue_mapfile, index=False)

    # Left Join mdf_filtered and tdf_new on Tissue and write to file
    mdf_new = mdf_filtered.merge(tdf, on="Tissue",
                                 how="left").drop(["Count"], axis=1)
    mdf_new = mdf_new.merge(tdf_new, how="left",
                                 on="Tissue").drop(["Count"], axis=1)
    # mdf_new = mdf_filtered.merge(tdf_new, how="left",
    #                              on="Tissue").drop(["Count"], axis=1)

    mdf_new = mdf_new.rename(columns={"Sample":"SampleID"})
    mdf_new = mdf_new.rename(columns={"Tissue.1":"TissueClean"})
    print(mdf_new.shape)
    print(mdf_new.columns)

    mdf_new["VegetativeRepro"].replace(to_replace="Root",
                                        value="Vegetative", inplace=True)
    mdf_new["VegetativeRepro"].replace(to_replace="Hypotocyl",
                                        value="Vegetative", inplace=True)

    mdf_new["AboveBelow"].replace(to_replace="Seed",
                                        value="WholePlant", inplace=True)

    print(f'AboveBelow Classes: {mdf_new["AboveBelow"].value_counts()}')
    print(f'VegetativeRepro Classes:\
          {mdf_new["VegetativeRepro"].value_counts()}')

    mdf_new.to_csv(filtered_metafile, sep=",", index=False)

    return None



if __name__ == "__main__":
    print("Begining metadata cleaning and filtering...")
    clean_metadata()
    print("Meatadata cleaning and filtering done")