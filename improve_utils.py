import os
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
from math import sqrt
from scipy import stats
from typing import List, Union, Optional, Tuple


fdir = Path(__file__).resolve().parent
#print(fdir)

# -----------------------------------------------------------------------------
# TODO
# Note!
# We need to decide how this utils file will be provided for each model.
# Meanwhile, place this .py file in the level as your data preprocessing script.
# For example:
# GraphDRP/
# |_______ preprocess.py
# |_______ improve_utils.py
# |
# | 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Global variables
# ----------------
# These are globals for all models
import types
improve_globals = types.SimpleNamespace()

# TODO:
# This is CANDLE_DATA_DIR (or something...).
# How this is going to be passed to the code?
improve_globals.main_data_dir = fdir/"csa_data"
# improve_globals.main_data_dir = fdir/"improve_data_dir"
# imp_globals.main_data_dir = fdir/"candle_data_dir"

# Dir names corresponding to the primary input/output blocks in the pipeline
# {}: input/output
# []: process
# train path:      {raw_data} --> [preprocess] --> {ml_data} --> [train] --> {models}
# inference path:  {ml_data, models} --> [inference] --> {infer}
improve_globals.raw_data_dir_name = "raw_data"  # benchmark data
improve_globals.ml_data_dir_name = "ml_data"    # preprocessed data for a specific ML model
improve_globals.models_dir_name = "models"      # output from model training
improve_globals.infer_dir_name = "infer"        # output from model inference (testing)

# Secondary dirs in raw_data
improve_globals.x_data_dir_name = "x_data"      # feature data
improve_globals.y_data_dir_name = "y_data"      # target data
improve_globals.splits_dir_name = "splits"      # splits files

# Column names in the raw data files
# imp_globals.canc_col_name = "CancID"
# imp_globals.drug_col_name = "DrugID"
improve_globals.canc_col_name = "improve_sample_id"  # column name that contains the cancer sample ids TODO: rename to sample_col_name
improve_globals.drug_col_name = "improve_chem_id"    # column name that contains the drug ids
improve_globals.source_col_name = "source"           # column name that contains source/study names (CCLE, GDSCv1, etc.)
improve_globals.pred_col_name_suffix = "_pred"       # suffix to predictions col name (example of final col name: auc_pred)

# Response data file name
improve_globals.y_file_name = "response.tsv"  # response data

# Cancer sample features file names
improve_globals.copy_number_fname = "cancer_copy_number.tsv"  # cancer feature
improve_globals.discretized_copy_number_fname = "cancer_discretized_copy_number.tsv"  # cancer feature
improve_globals.dna_methylation_fname = "cancer_DNA_methylation.tsv"  # cancer feature
improve_globals.gene_expression_fname = "cancer_gene_expression.tsv"  # cancer feature
improve_globals.miRNA_expression_fname = "cancer_miRNA_expression.tsv"  # cancer feature
improve_globals.mutation_count_fname = "cancer_mutation_count.tsv"  # cancer feature
improve_globals.mutation_fname = "cancer_mutation.tsv"  # cancer feature
improve_globals.rppa_fname = "cancer_RPPA.tsv"  # cancer feature

# Drug features file names
improve_globals.smiles_file_name = "drug_SMILES.tsv"  # drug feature
improve_globals.mordred_file_name = "drug_mordred.tsv"  # drug feature
improve_globals.ecfp4_512bit_file_name = "drug_ecfp4_nbits512.tsv"  # drug feature
improve_globals.cell_mutation_fname = "cancer_mutation_count.tsv" #cancer feather

# Globals derived from the ones defined above
improve_globals.raw_data_dir = improve_globals.main_data_dir/improve_globals.raw_data_dir_name # raw_data
improve_globals.ml_data_dir  = improve_globals.main_data_dir/improve_globals.ml_data_dir_name  # ml_data
improve_globals.models_dir   = improve_globals.main_data_dir/improve_globals.models_dir_name   # models
improve_globals.infer_dir    = improve_globals.main_data_dir/improve_globals.infer_dir_name    # infer
# -----
improve_globals.x_data_dir   = improve_globals.raw_data_dir/improve_globals.x_data_dir_name    # x_data
improve_globals.y_data_dir   = improve_globals.raw_data_dir/improve_globals.y_data_dir_name    # y_data
improve_globals.splits_dir   = improve_globals.raw_data_dir/improve_globals.splits_dir_name    # splits

# Response
improve_globals.y_file_path = improve_globals.y_data_dir/improve_globals.y_file_name           # response.txt

# Cancers
improve_globals.copy_number_file_path = improve_globals.x_data_dir/improve_globals.copy_number_fname  # cancer_copy_number.txt
improve_globals.discretized_copy_number_file_path = improve_globals.x_data_dir/improve_globals.discretized_copy_number_fname # cancer_discretized_copy_number.txt
improve_globals.dna_methylation_file_path = improve_globals.x_data_dir/improve_globals.dna_methylation_fname  # cancer_DNA_methylation.txt
improve_globals.gene_expression_file_path = improve_globals.x_data_dir/improve_globals.gene_expression_fname  # cancer_gene_expression.txt
improve_globals.mirna_expression_file_path = improve_globals.x_data_dir/improve_globals.miRNA_expression_fname  # cancer_miRNA_expression.txt
improve_globals.mutation_count_file_path = improve_globals.x_data_dir/improve_globals.mutation_count_fname # cancer_mutation_count.txt
improve_globals.mutation_file_path = improve_globals.x_data_dir/improve_globals.mutation_fname # cancer_mutation.txt
improve_globals.rppa_file_path = improve_globals.x_data_dir/improve_globals.rppa_fname # cancer_RPPA.txt

# Drugs
improve_globals.smiles_file_path = improve_globals.x_data_dir/improve_globals.smiles_file_name  # 
improve_globals.mordred_file_path = improve_globals.x_data_dir/improve_globals.mordred_file_name  # 
improve_globals.ecfp4_512bit_file_path = improve_globals.x_data_dir/improve_globals.ecfp4_512bit_file_name  # 
improve_globals.cell_mutation_file_path = improve_globals.x_data_dir/improve_globals.cell_mutation_fname
# -----------------------------------------------------------------------------


# -------------------------------------
# Drug response loaders
# -------------------------------------

def load_cell_mutation_data(
        gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
        sep: str="\t", verbose: bool=True) -> pd.DataFrame:
    """
    Returns gene expression data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.cell_mutation_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"cell mutation data: {df.shape}")
        # print(df.dtypes)
        # print(df.dtypes.value_counts())
    return df


def load_single_drug_response_data(
    # source: Union[str, List[str]],
    source: str,
    split: Union[int, None]=None,
    split_type: Union[str, List[str], None]=None,
    y_col_name: str="auc",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns datarame with cancer ids, drug ids, and drug response values. Samples
    from the original drug response file are filtered based on the specified
    sources.

    Args:
        source (str or list of str): DRP source name (str) or multiple sources (list of strings)
        split(int or None): split id (int), None (load all samples)
        split_type (str or None): one of the following: 'train', 'val', 'test'
        y_col_name (str): name of drug response measure/score (e.g., AUC, IC50)

    Returns:
        pd.Dataframe: dataframe that contains drug response values
    """
    # TODO: at this point, this func implements the loading a single source
    df = pd.read_csv(improve_globals.y_file_path, sep=sep)

    # import pdb; pdb.set_trace()
    if isinstance(split, int):
        # Get a subset of samples
        ids = load_split_file(source, split, split_type)
        df = df.loc[ids]
    else:
        # Get the full dataset for a given source
        df = df[df[improve_globals.source_col_name].isin([source])]

    cols = [improve_globals.source_col_name,
            improve_globals.drug_col_name,
            improve_globals.canc_col_name,
            y_col_name]
    df = df[cols]  # [source, drug id, cancer id, response]
    df = df.reset_index(drop=True)
    if verbose:
        print(f"Response data: {df.shape}")
        print(df[[improve_globals.canc_col_name, improve_globals.drug_col_name]].nunique())
    return df


def load_single_drug_response_data_v2(
    # source: Union[str, List[str]],
    source: str,
    # split: Union[int, None]=None,
    # split_type: Union[str, List[str], None]=None,
    split_file_name: Union[str, List[str], None]=None,
    y_col_name: str="auc",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns datarame with cancer ids, drug ids, and drug response values. Samples
    from the original drug response file are filtered based on the specified
    sources.

    Args:
        source (str or list of str): DRP source name (str) or multiple sources (list of strings)
        split(int or None): split id (int), None (load all samples)
        split_type (str or None): one of the following: 'train', 'val', 'test'
        y_col_name (str): name of drug response measure/score (e.g., AUC, IC50)

    Returns:
        pd.Dataframe: dataframe that contains drug response values
    """
    # TODO: currently, this func implements loading a single data source (CCLE or CTRPv2 or ...)
    df = pd.read_csv(improve_globals.y_file_path, sep=sep)

    # Get a subset of samples
    if isinstance(split_file_name, list) and len(split_file_name) == 0:
        raise ValueError("Empty list is passed via split_file_name.")
    if isinstance(split_file_name, str):
        split_file_name = [split_file_name]
    ids = load_split_ids(split_file_name)
    df = df.loc[ids]
    # else:
    #     # Get the full dataset for a given source
    #     df = df[df[improve_globals.source_col_name].isin([source])]

    # # Get a subset of cols
    # cols = [improve_globals.source_col_name,
    #         improve_globals.drug_col_name,
    #         improve_globals.canc_col_name,
    #         y_col_name]
    # df = df[cols]  # [source, drug id, cancer id, response]

    df = df.reset_index(drop=True)
    if verbose:
        print(f"Response data: {df.shape}")
        print(f"Unique cells:  {df[improve_globals.canc_col_name].nunique()}")
        print(f"Unique drugs:  {df[improve_globals.drug_col_name].nunique()}")
    return df


def load_split_ids(split_file_name: Union[str, List[str]]) -> List[int]:
    """ Returns list of integers, representing the rows in the response dataset.
    Args:
        split_file_name (str or list of str): splits file name or list of file names

    Returns:
        list: list of integers representing the ids
    """
    ids = []
    for fname in split_file_name:
        fpath = improve_globals.splits_dir/fname
        assert fpath.exists(), f"split_file_name {fname} not found."
        ids_ = pd.read_csv(fpath, header=None)[0].tolist()
        ids.extend(ids_)
    return ids


def load_split_file(
    source: str,
    split: Union[int, None]=None,
    split_type: Union[str, List[str], None]=None) -> List[int]:
    """
    Args:
        source (str): DRP source name (str)

    Returns:
        ids (list): list of id integers
    """
    # TODO: used in the old version of the rsp loader
    if isinstance(split_type, str):
        split_type = [split_type]

    # Check if the split file exists and load
    ids = []
    for st in split_type:
        fpath = improve_globals.splits_dir/f"{source}_split_{split}_{st}.txt"
        assert fpath.exists(), f"Splits file not found: {fpath}"
        ids_ = pd.read_csv(fpath, header=None)[0].tolist()
        ids.extend(ids_)
    return ids


# -------------------------------------
# Omic feature loaders
# -------------------------------------

"""
Notes about omics data.

Omics data files are multi-level tables with several column types (generally 3
or 4), each contains gene names using a different gene identifier system:
Entrez ID, Gene Symbol, Ensembl ID, TSS

The column levels are not organized in the same order across the different
omic files.

The level_map dict, in each loader function, encodes the column level and the
corresponding identifier systems.

For example, in the copy number file the level_map is:  
level_map = {"Entrez":0, "Gene_Symbol": 1, "Ensembl": 2}
"""

def set_col_names_in_multilevel_dataframe(
    df: pd.DataFrame,
    level_map: dict,
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol") -> pd.DataFrame:
    """ Util function that supports loading of the omic data files.
    Returns the input dataframe with the multi-level column names renamed as
    specified by the gene_system_identifier arg.

    Args:
        df (pd.DataFrame): omics dataframe
        level_map (dict): encodes the column level and the corresponding identifier systems
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: the input dataframe with the specified multi-level column names
    """
    df = df.copy()

    level_names = list(level_map.keys())
    level_values = list(level_map.values())
    n_levels = len(level_names)
    
    if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
        gene_system_identifier = gene_system_identifier[0]

    # print(gene_system_identifier)
    # import pdb; pdb.set_trace()
    if isinstance(gene_system_identifier, str):
        if gene_system_identifier == "all":
            df.columns = df.columns.rename(level_names, level=level_values)  # assign multi-level col names
        else:
            df.columns = df.columns.get_level_values(level_map[gene_system_identifier])  # retian specific column level
    else:
        assert len(gene_system_identifier) <= n_levels, f"'gene_system_identifier' can't contain more than {n_levels} items."
        set_diff = list(set(gene_system_identifier).difference(set(level_names)))
        assert len(set_diff) == 0, f"Passed unknown gene identifiers: {set_diff}"
        kk = {i: level_map[i] for i in level_map if i in gene_system_identifier}
        # print(list(kk.keys()))
        # print(list(kk.values()))
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())  # assign multi-level col names
        drop_levels = list(set(level_map.values()).difference(set(kk.values())))
        df = df.droplevel(level=drop_levels, axis=1)
    return df


def load_copy_number_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns copy number data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.copy_number_file_path, sep=sep, index_col=0, header=header)
    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    # Test the func
    # d0 = set_col_names_in_multilevel_dataframe(df, "all")
    # d1 = set_col_names_in_multilevel_dataframe(df, "Ensembl")
    # d2 = set_col_names_in_multilevel_dataframe(df, ["Ensembl"])
    # d3 = set_col_names_in_multilevel_dataframe(df, ["Entrez", "Gene_Symbol", "Ensembl"])
    # d4 = set_col_names_in_multilevel_dataframe(df, ["Entrez", "Ensembl"])
    # d5 = set_col_names_in_multilevel_dataframe(df, ["Blah", "Ensembl"])
    if verbose:
        print(f"Copy number data: {df.shape}")
        # print(df.dtypes)
        # print(df.dtypes.value_counts())
    return df


def load_discretized_copy_number_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns discretized copy number data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.discretized_copy_number_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"Discretized copy number data: {df.shape}")

    return df


def load_dna_methylation_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns methylation data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    level_map = {"Ensembl": 2, "Entrez": 1, "Gene_Symbol": 3, "TSS": 0}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.dna_methylation_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"DNA methylation data: {df.shape}")
        # print(df.dtypes)  # TODO: many column are of type 'object'
        # print(df.dtypes.value_counts())
    return df


def load_gene_expression_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns gene expression data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.gene_expression_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"Gene expression data: {df.shape}")
    return df


def load_mirna_expression_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    # TODO
    raise NotImplementedError("The function is not implemeted yet.")
    return None


def load_mutation_count_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns mutation count data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.mutation_count_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"Mutation count data: {df.shape}")
    
    return df


def load_mutation_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    # TODO
    raise NotImplementedError("The function is not implemeted yet.")
    return None


def load_rppa_data(
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    # TODO
    raise NotImplementedError("The function is not implemeted yet.")
    return None




# -------------------------------------
# Drug feature loaders
# -------------------------------------

def load_smiles_data(
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    df = pd.read_csv(improve_globals.smiles_file_path, sep=sep)

    # TODO: updated this after we update the data
    df.columns = ["improve_chem_id", "smiles"]

    if verbose:
        print(f"SMILES data: {df.shape}")
        # print(df.dtypes)
        # print(df.dtypes.value_counts())
    return df


def load_mordred_descriptor_data(
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Return Mordred descriptors data.
    """
    df = pd.read_csv(improve_globals.mordred_file_path, sep=sep)
    df = df.set_index(improve_globals.drug_col_name)
    if verbose:
        print(f"Mordred descriptors data: {df.shape}")
    return df


def load_morgan_fingerprint_data(
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Return Morgan fingerprints data.
    """
    df = pd.read_csv(improve_globals.ecfp4_512bit_file_path, sep=sep)
    df = df.set_index(improve_globals.drug_col_name)
    return df


# -------------------------------------
# Save data functions
# -------------------------------------

def save_preds(df: pd.DataFrame, y_col_name: str,
               outpath: Union[str, PosixPath], round_decimals: int=4) -> None:
    """ Save model predictions.
    This function throws errors if the dataframe does not include the expected
    columns: canc_col_name, drug_col_name, y_col_name, y_col_name + "_pred"

    Args:
        df (pd.DataFrame): df with model predictions
        y_col_name (str): drug response col name (e.g., IC50, AUC)
        outpath (str or PosixPath): outdir to save the model predictions df
        round (int): round response values 
        
    Returns:
        None
    """
    # Check that the 4 columns exist
    assert improve_globals.canc_col_name in df.columns, f"{improve_globals.canc_col_name} was not found in columns."
    assert improve_globals.drug_col_name in df.columns, f"{improve_globals.drug_col_name} was not found in columns."
    assert y_col_name in df.columns, f"{y_col_name} was not found in columns."
    pred_col_name = y_col_name + f"{improve_globals.pred_col_name_suffix}"
    assert pred_col_name in df.columns, f"{pred_col_name} was not found in columns."

    # Round
    df = df.round({y_col_name: round_decimals, pred_col_name: round_decimals})

    # Save preds df
    df.to_csv(outpath, index=False)
    return None






# ==================================================================
# Leftovers
# ==================================================================
def get_data_splits(
    src_raw_data_dir: str,
    splitdir_name: str,
    split_file_name: str,
    rsp_df: pd.DataFrame):
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    splitdir = src_raw_data_dir/splitdir_name
    if len(split_file_name) == 1 and split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
    else:
        # Check if the split file exists and load
        ids = []
        for fname in split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)

    """
    # Method 1
    splitdir = Path(os.path.join(src_raw_data_dir))/"splits"
    if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
        outdir_name = "full"
    else:
        # Check if the split file exists and load
        ids = []
        split_id_str = []    # e.g. split_5
        split_type_str = []  # e.g. tr, vl, te
        for fname in args.split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                # Get the ids
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)
                # Get the name
                fname_sep = fname.split("_")
                split_id_str.append("_".join([s for s in fname_sep[:2]]))
                split_type_str.append(fname_sep[2])
        assert len(set(split_id_str)) == 1, "Data splits must be from the same dataset source."
        split_id_str = list(set(split_id_str))[0]
        split_type_str = "_".join([x for x in split_type_str])
        outdir_name = f"{split_id_str}_{split_type_str}"
    ML_DATADIR = main_data_dir/"ml_data"
    root = ML_DATADIR/f"data.{args.source_data_name}"/outdir_name # ML data
    os.makedirs(root, exist_ok=True)
    """

    """
    # Method 2
    splitdir = src_raw_data_dir/args.splitdir_name
    if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
    else:
        # Check if the split file exists and load
        ids = []
        for fname in args.split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)
    """
    return ids


def get_common_samples(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ref_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        df1, df2 (pd.DataFrame): dataframes
        ref_col (str): the ref column to find the common values

    Returns:
        df1, df2

    Example:
        TODO
    """
    # Retain (canc, drug) response samples for which we have omic data
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    # print(df1.shape)
    df1 = df1[ df1[improve_globals.canc_col_name].isin(common_ids) ].reset_index(drop=True)
    # print(df1.shape)
    # print(df2.shape)
    df2 = df2[ df2[improve_globals.canc_col_name].isin(common_ids) ].reset_index(drop=True)
    # print(df2.shape)
    return df1, df2


def read_df(fpath: str, sep: str=","):
    """
    IMPROVE-specific func.
    Load a dataframe. Supports csv and parquet files.
    sep : the sepator in the csv file
    """
    # TODO: this func might be available in candle
    assert Path(fpath).exists(), f"File {fpath} was not found."
    if "parquet" in str(fpath):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, sep=sep)
    return df


def get_subset_df(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    """ Get a subset of the input dataframe based on row ids."""
    df = df.loc[ids]
    return df


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def r_square(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)
