import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

# additional definitions
additional_definitions = [
    {
        "name": "gene2id",
        "type": str,
        "help": "path to gene2id file",
    },
    {   
        "name": "drug2id",
        "type": str,
        "help": "path to drug to ID file",
    },
    {
        "name": "cell2id",
        "type": str,
        "help": "Path to cell 2 id file",
    },
    {   
        "name": "genotype_hiddens",
        "type": int,
        "help": "Number of hidden genotypes",
    },
    {   
        "name": "drug_hiddens",
        "type": str, 
        "help": "string of values to indicate the number of hidden layers",
    },
    {   
        "name": "test_size",
        "type": float, 
        "help": "test data size",
    },
    {   
        "name": "val_size",
        "type": float, 
        "help": "validation data size",
    },
    {  
        "name": "n_fold",
        "type": int, 
        "help": "number of folds in cross-validation",
    },
    {  
        "name": "regularization",
        "type": int,
        "help": "L2 regularizations of weights and biases",
    },
    {  
        "name": "ppi_data",
        "type": str,
        "help": "path to protein-protein interaction (PPI) network data",
    },
    {   
        "name": "response_data",
        "type": str,
        "help": "path to drug response data",
    },
    {   
        "name": "gene_data",
        "type": str, 
        "help": "path to gene expression data",
    },  
    {   
        "name": "levels",
        "type": int, 
        "help": "number of coarsened graphs",
    },    
]

# required definitions
required = [
    "predict",
    "genotype",
    "fingerprint",
]

# initialize class
class DrugGCN(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions",

