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

# IMPROVE PROJECT INSTRUCTIONS

The improve project requires standarized interfaces for data preprocessing, training and inference

# DATA PREPROCESSING

To create the data run the preprocess.sh code to download the data. To use a custom dataset, set the 'improve_analysis" flag to 'yes' in the DrugCell_params.txt file

# Model Training

1. train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR 

CANDLE_DATA_DIR=<PATH OF REPO/Data/>

Note: The train.sh script will download the original authors data if the Data directory is empty

      * set CUDA_VISIBLE_DEVICES to a GPU device ID to make this devices visible to the application.
      * CANDLE_DATA_DIR, path to base CANDLE directory for model input and outputs.
      * CANDLE_CONFIG , path to CANDLE config file must be inside CANDLE_DATA_DIR.

## Example

   * git clone ....
   * cd DrugCell
   * mkdir Data	
   * check permissions if all scripts are executable
   * ./preprocess.sh 2 ./Data
   * ./train.sh 2 ./Data
   * ./infer.sh 2 ./Data


## Setting up environment

This model is curated as part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE)

### Install Conda version version 22.11.1

* step 1: conda create -n drugcell_python python=3.9.15 anaconda
* step 2: conda activate drucell_python
* step 3: conda env update --name drugcell_python --file environment.yml
* step 4: pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 torchmetrics==0.11.1 --extra-index-url https://download.pytorch.org/whl/cu113
* step 5: pip install git+https://github.com/ECP-CANDLE/candle_lib@0d32c6bb97ace0370074194943dbeaf9019e6503


# DrugCell release v1.0
DrugCell v1.0 was trained using (cell line, drug) pairs, but 
it can be generalized to estimate response of any cells to any drugs if:
1. The feature vector of cell is built as a binary vector representing 
mutational status of 3,008 genes (the list of index and name of the genes 
is provided in _gene2ind.txt_). 
2. The feature vector of drug is encoded into a binary vector of length 2,048 
using Morgan Fingerprint (radius = 2). We also provide the pre-computed 
feature vectors for 684 drugs in our training data (_drug2fingerprint.txt_).

**Pre-trained DrugCell v1.0 model and the drug response data for 509,294 (cell line, drug) pairs used to train the model is shared in http://drugcell.ucsd.edu/downloads.**

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
    * _drug2fingerprint.txt_: a comma-delimited file where each row has 2,048 binary values which would form 
    , when combined, a Morgan Fingerprint representation of each drug. 
    The line number of should match with the indices of drugs in _drug2ind.txt_ file. 
3. Test data file: _drugcell_test.txt_
    * A tab-delimited file containing all data points that you want to estimate drug response for. 
    The 1st column is identification of cells (genotypes) and the 2nd column is identification of 
    drugs.
    
To load a pre-trained model used for analyses in our manuscript and make prediction for (cell, drug) pairs of 
your interest, execute the following:

1. Make sure you have _gene2ind.txt_, _cell2ind.txt_, _cell2mutation.txt_, _drug2ind.txt_, 
_drug2fingerprint.txt_, and your file containing test data in proper format (examples are provided in 
_data_ and _sample_ folder)


1. Cell feature files: _gene2ind.txt_, _cell2ind.txt_, _cell2mutation.txt_
    * A detailed description about the contents of the files is given in _DrugCell release v1.0_ section.
   
2. Drug feature files: _drug2ind.txt_, _drug2fingerprints.txt_
    * A detailed description about the contents of the files is given in _DrugCell release v1.0_ section.

3. Training data file: _drugcell_train.txt_
    * A tab-delimited file containing all data points that you want to use to train the model. 
    The 1st column is identification of cells (genotypes), the 2nd column is identification of 
    drugs and the 3rd column is an observed drug response in a floating number. The current 
    version of the DrugCell code utilizes a loss function better suited for a regression problem (Minimum Squared Error; MSE), 
    and we recommend using the code to train a regressor rather a classifier. 
  
4. Validation data file: _drugcell_val.txt_
    * A tab-delimited file that in the same format as the training data. DrugCell training 
    script would evaluate the model trained in each iteration using the data contained 
    in this file. The performance of the model on the validation data may be used 
    as an early termination condition.
    
5. Ontology (hierarchy) file: _drugcell_ont.txt_
    * A tab-delimited file that contains the ontology (hierarchy) that defines the structure of a branch 
    of a DrugCell model that encodes the genotypes. The first column is always a term (subsystem or pathway), 
    and the second column is a term or a gene. 
    The third column should be set to "default" when the line represents a link between terms, 
    "gene" when the line represents an annotation link between a term and a gene. 
    The following is an example describing a sample hierarchy.
    
        ![](https://github.com/idekerlab/DrugCell/blob/master/misc/drugcell_ont_image_sample.png)

    ```
     GO:0045834	GO:0045923	default
     GO:0045834	GO:0043552	default
     GO:0045923	AKT2	gene
     GO:0045923	IL1B	gene
     GO:0043552	PIK3R4	gene
     GO:0043552	SRC	gene
     GO:0043552	FLT1	gene       
    ```
        
     * Example of the file (_drugcell_ont.txt_) is provided in _data_ folder.    

     
There are a few optional parameters that you can provide in addition to the input files:

1. _-model_: a name of directory where you want to store the trained models. The default 
is set to "MODEL" in the current working directory.

2. _-genotype_hiddens_: a number of neurons to assign each subsystem in the hierarchy. 
The default is set to 6. 

3. _-drug_hiddens_: a string listing the number of neurons for the drug-encoding branch 
of DrugCell. The number should be delimited by comma. The default value is "100,50,6", 
and with the default option, 
the drug branch of the resulting DrugCell model will be a fully-connected neural network with 3 layers 
consisting of 100, 50, and 6 neurons. 

4. _-final_hiddens_: the number of neurons in the top layer of DrugCell that combines 
the genotype-encoding and the drug-encoding branches. The default is 6.

5. _-epoch_: the number of epoch to run during the training phase. The default is set to 300.

6. _-batchsize_: the size of each batch to process at a time. The deafult is set to 5000. 
You may increase this number to speed up the training process within the memory capacity 
of your GPU server.

7. _-cuda_: the ID of GPU unit that you want to use for the model training. The default setting 
is to use GPU 0. 

Finally, to train a DrugCell model, execute a command line similar to the example provided in 
_sample/commandline_cuda.sh_:


# Example data files in _sample_ directory
There are three subsets of our training data provided as toy example: drugcell_train.txt, drugcell_test.txt and drugcell_val.txt have 10,000, 1,000, and 1,000 (cell line, drug) pairs along with the corresponding drug response (area under the dose-response curve). 

