# DrugCell: a visible neural network model for drug response prediction 
DrugCell is an interpretable neural network-based model that predicts 
cell response to a wide range of drugs. Unlike fully-connected neural networks,
connectivity of neurons in the DrugCell mirrors a 
biological hierarchy (e.g. Gene Ontology), so that the information travels 
only between subsystems (or pathways) with known hierarchical relationship 
during the model training. 
This feature of the framework allows for identification of 
subsystems in the hierarchy that are important to the model's prediction, 
warranting further investigation on underlying biological mechanisms of 
cell response to treatments. 

The current version (v1.0) of the DrugCell model 
is trained using 509,294 (cell line, drug) pairs across 
1,235 tumor cell lines and 684 drugs. The training data is retrieved from Genomics of
Drug Sensitivity in Cancer database (GDSC) and the Cancer Therapeutics Response 
Portal (CTRP) v2. 

DrugCell characterizes each cell line using its genotype; 
the feature vector for each cell is a binary vector representing 
mutational status of the top 15% most frequently mutated genes (n = 3,008) 
in cancer. 
Drugs are encoded using Morgan Fingerprint (radius = 2), and the resulting 
feature vectors are binary vectors of length 2,046. 

# Environment set up for training and testing of DrugCell v1.0 
DrugCell training/testing scripts require the following environmental setup:

* Hardware
    * GPU server with CUDA>=10 installed

* Software
    * Python v2.7
    * PyTorch
    * numpy
    * networkx 
    * A virtual environment to run model training/testing can be created using _environment_setup/environment.yml_ file
        * conda env create -f _environment.yml_
    * After setting up the conda virtual environment, make sure to activate environment before executing DrugCell scripts.
        * source activate pytorch3drugcell


# DrugCell release v1.0
DrugCell v1.0 was trained using (cell line, drug) pairs, but 
it can be generalized to estimate response of any cells to any drugs if:
1. The feature vector of cell is built as a binary vector representing 
mutational status of 3,008 genes (the list of index and name of the genes 
is provided in _gene2ind.txt_). 
2. The feature vector of drug is encoded into a binary vector of length 2,046 
using Morgan Fingerprint (radius = 2). We also provide the pre-computed 
feature vectors for 684 drugs in our training data (_drug2fingerprint.txt_).

Required input files:
1. Cell feature files: _gene2ind.txt_, _cell2ind.txt_, _cell2mutation.txt_
    * _gene2ind.txt_: make sure you are using _gene2ind.txt_ file provided in this repository.
    * _cell2ind.txt_: a tab-delimited file where the 1st column is index of cells and the 2nd column is the name of cells (genotypes).
    * _cell2mutation.txt_: a comma-delimited file where each row has 3,008 binary values indicating each gene is mutated (1) or not (0). 
    The column index of each gene should match with those in _gene2ind.txt_ file. The line number should 
    match with the indices of cells in _cell2ind.txt_ file.
2. Drug feature files: drug2ind, drug2fingerprints
    * _drug2ind.txt_: a tab-delimited file where the 1st column is index of drug and the 2nd column is 
    identification of each drug (e.g., SMILES representation or name). The identification of drugs 
    should match to those in _drug2fingerprint.txt_ file. 
    * _drug2fingerprint.txt_: a comma-delimited file where each row has 2,046 binary values which would form 
    , when combined, a Morgan Fingerprint representation of each drug. 
    The line number of should match with the indices of drugs in _drug2ind.txt_ file. 
3. Test data file: testdata.txt
    * A tab-delimited file containing all data points that you want to estimate drug response for. 
    The 1st column is identification of cells (genotypes) and the 2nd column is identification of 
    drugs.
    
To load a pre-trained model used for analyses in our manuscript and make prediction for (cell, drug) pairs of 
your interest, execute the following:
1. Make sure you have _gene2ind.txt_, _cell2ind.txt_, _cell2mutation.txt_, _drug2ind.txt_, 
_drug2fingerprint.txt_, and your file containing test data in proper format (examples are provided in 
_data_ and _sample_ folder)

2. To load it in a cpu server, execute the following:
    ```
    python predict_drugcell_cpu.py -gene2id _gene2ind.txt_ 
                                   -cell2id _cell2ind.txt_ 
                                   -drug2id _drug2ind.txt_ 
                                   -cellline _cell2mutation.txt_ 
                                   -fingerprint _drug2fingerprint.txt_ 
                                   -predict _testdata.txt_ 
                                   -hidden _<path_to_directory_to_store_hidden_values>_ 
                                   -result _<path_to_directory_to_store_prediction_results>_ 
                                   -load _<path_to_model_file>_
    ```
    * A bash script is provided in _sample_ folder as a specific example. 

3. To run the model in a GPU server, run predict_drugcell.py (instead of predict_drugcell_cpu.py) 
with same set of parameters as 2.




      