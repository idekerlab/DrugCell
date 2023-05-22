=================
DrugCell
=================
A visible neural network model for drug response prediction

---------
Structure
---------
DrugCell is an interpretable neural network-based model that predicts cell response to a wide range of drugs. Unlike fully-connected neural networks, connectivity of neurons in the DrugCell mirrors a biological hierarchy (e.g. Gene Ontology), so that the information travels only between subsystems (or pathways) with known hierarchical relationship during the model training. This feature of the framework allows for identification of subsystems in the hierarchy that are important to the model's prediction, warranting further investigation on underlying biological mechanisms of cell response to treatments.

The current version (v1.0) of the DrugCell model is trained using 509,294 (cell line, drug) pairs across 1,235 tumor cell lines and 684 drugs. The training data is retrieved from Genomics of Drug Sensitivity in Cancer database (GDSC) and the Cancer Therapeutics Response Portal (CTRP) v2.

DrugCell characterizes each cell line using its genotype; the feature vector for each cell is a binary vector representing mutational status of the top 15% most frequently mutated genes (n = 3,008) in cancer. Drugs are encoded using Morgan Fingerprint (radius = 2), and the resulting feature vectors are binary vectors of length 2,048.

----
Data
----

Data sources
------------
The primary data sources that have been used to construct ML datasets include:

- IMPROVE CCLE
- PubChem (drug SMILES)

Raw data
--------
Data location:  https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/
The data includes omic and drug features (x_data folder), response data (y_data folder), and data splits (splits folder).

ML data
-------
The script `preprocess_new.py >`__ uses raw data to generate ML data that can be used to train and test with DrugCell. The necessary raw data are automatically downloaded from the FTP server using a `candle_lib` utility function `get_file()` and processed:

- **Response data**. AUC values
- **Cancer features**. mutation data
- **Drug features**. SMILES string 

----------
Evaluation
----------

----
URLs
----


----------
References
----------
Kuenzi BM, Park J, Fong SH, Sanchez KS, Lee J, Kreisberg JF, Ma J, Ideker T. Predicting Drug Response and Synergy Using a Deep Learning Model of Human Cancer Cells. Cancer Cell. 2020 Nov 9;38(5):672-684.e6. doi: 10.1016/j.ccell.2020.09.014. Epub 2020 Oct 22. PMID: 33096023; PMCID: PMC7737474.
