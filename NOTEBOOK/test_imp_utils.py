import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional

import improve_utils
from improve_utils import improve_globals as ig


fdir = Path(__file__).resolve().parent

# Settings:
y_col_name = "auc"

#import pdb; pdb.set_trace()
# Load drug response data
rs = improve_utils.load_single_drug_response_data(source="CCLE", y_col_name=y_col_name)  # load all samples
rs = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type=["train", "val"])  # combine train and val sets
# ------
# rs_tr = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type="train", y_col_name=y_col_name)  # load train
# rs_vl = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type="val", y_col_name=y_col_name)    # load val
# rs_te = improve_utils.load_single_drug_response_data(source="CCLE", split=0, split_type="test", y_col_name=y_col_name)   # load test
# ------
split = 0
source_data_name = "CCLE"

# Load train
rs_tr = improve_utils.load_single_drug_response_data_v2(
    source=source_data_name,
    split_file_name=f"{source_data_name}_split_{split}_train.txt",
    y_col_name=y_col_name)

# Load val
rs_vl = improve_utils.load_single_drug_response_data_v2(
    source=source_data_name,
    split_file_name=f"{source_data_name}_split_{split}_val.txt",
    y_col_name=y_col_name)

# Load test
rs_te = improve_utils.load_single_drug_response_data_v2(
    source=source_data_name,
    split_file_name=f"{source_data_name}_split_{split}_test.txt",
    y_col_name=y_col_name)

print("\nResponse train data", rs_tr.shape)
print("Response val data", rs_vl.shape)
print("Response test data", rs_te.shape)

# Load omic feature data
# cv = improve_utils.load_copy_number_data(gene_system_identifier="Gene_Symbol")
ge = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
# mt = improve_utils.load_dna_methylation_data(gene_system_identifier="TSS")

# Load drug feature data
sm = improve_utils.load_smiles_data()
dd = improve_utils.load_mordred_descriptor_data()
fp = improve_utils.load_morgan_fingerprint_data()

# import pdb; pdb.set_trace()
print(f"Total unique cells: {rs_tr[ig.canc_col_name].nunique()}")
print(f"Total unique drugs: {rs_tr[ig.drug_col_name].nunique()}")
assert len(set(rs_tr[ig.canc_col_name]).intersection(set(ge.index))) == rs_tr[ig.canc_col_name].nunique(), "Something is missing..."
assert len(set(rs_tr[ig.drug_col_name]).intersection(set(fp.index))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."

print("Finished.")
