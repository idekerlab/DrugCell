import os
import candle
import torch
import torchvision
import numpy as np
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from predict_drugcell import main
from utils.util import load_mapping
from utils.util import load_train_data
from utils.util import build_input_vector
from utils.util import pearson_corr

file_path = os.path.dirname(os.path.realpath(__file__))
print(file_path)

# Just because the tensorflow warnings are a bit verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'


# additional definitions
additional_definitions = [
    {
        "name": "batchsize",
        "type": int,
        "help": "...",
    },
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
        "name": "hidden",
        "type": str, 
        "help": "string to indicate hidden output layer ",
    },
    {   
        "name": "cuda",
        "type": int, 
        "help": "CUDA ID",
    },
    {  
        "name": "result",
        "type": str, 
        "help": "result file name",
    },
]

# required definitions
required = [
    "genotype",
    "fingerprint",
]

# initialize class
class DrugCell_candle(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions


def initialize_parameters():
    preprocessor_bmk = DrugCell_candle(file_path,
        '../DrugCell_params.txt',
        'pytorch',
        prog='DrugCell_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

def load_mapping(map_file):
    mapping = {}
    with open(map_file) as fin:
        for raw_line in fin:
            line  = raw_line.strip().split()
            mapping[line[1]] = int(line[0])
    return mapping

def load_train_data(drug_data, cell2id_dict, drug2id_dict):
    data = []
    label = []
    with open(drug_data) as fin:
        for raw_line in fin:
            tokens = raw_line.strip().split('\t')
            data.append([cell2id_dict[tokens[0]], drug2id_dict[tokens[1]]])
            label.append([float(tokens[2])])
    return data, label


#def preprocess(params):
#    test_file = params['test_data']
#    train_file = params['train_data'] 
#    cell2id_mapping_file = params['cell2id']
#    drug2id_mapping_file = params['drug2id']
    
    # load mapping files
#    cell2id_mapping = load_mapping(cell2id_mapping_file)
#    drug2id_mapping = load_mapping(drug2id_mapping_file)
#    gene2id_mapping = load_mapping(params['gene2id'])
#    
#    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
#    train_feature, train_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
#    print('Total number of cell lines = %d' % len(cell2id_mapping))
#    print('Total number of drugs = %d' % len(drug2id_mapping))
#    cell_features = np.genfromtxt(params['genotype'], delimiter=',')
#    drug_features = np.genfromtxt(params['fingerprint'], delimiter=',')
#    num_cells = len(cell2id_mapping)
#    num_drugs = len(drug2id_mapping)
#    num_genes = len(gene2id_mapping)
#    drug_dim = len(drug_features[0,:])
#    return num_genes, drug_dim, cell_features, drug_features
    
def run(params):
    keys_parsing = ["train_data", "test_data", "val_data",
                    "onto", "genotype_hiddens", "fingerprint",
                    "genotype", "cell2id","drug2id", "drug_hiddens",
                    "model_name"]
#    candle.file_utils.get_file(params['original_data'], params['data_url'])
#    candle.file_utils.get_file(params['data_predict'], params['predict_url'])
#    candle.file_utils.get_file(params['data_model'], params['model_url'])
    print(os.environ['CANDLE_DATA_DIR'])
    data_download_filepath = candle.get_file(params['original_data'], params['data_url'],
                                        datadir = params['data_dir'],
                                        cache_subdir = None)
    print('download_path: {}'.format(data_download_filepath))
    predict_download_filepath = candle.get_file(params['data_predict'], params['predict_url'],
                                        datadir = params['data_dir'],
                                        cache_subdir = None)
    print('download_path: {}'.format(predict_download_filepath))
    model_download_filepath = candle.get_file(params['data_model'], params['model_url'],
                                        datadir = params['data_dir'],
                                        cache_subdir = None)
    print('download_path: {}'.format(model_download_filepath))

    model_param_key = []
    for key in params.keys():
        if key not in keys_parsing:
                model_param_key.append(key)
    model_params = {key: params[key] for key in model_param_key}
    params['model_params'] = model_params
    args = candle.ArgumentStruct(**params)
    cell2id_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['cell2id']
    drug2id_path  = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['drug2id']
    gene2id_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['gene2id']
    genotype_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['genotype']
    fingerprint_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['fingerprint']
    hidden_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['hidden']
    result_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['result']
    main(predict_download_filepath, cell2id_path, drug2id_path, str(gene2id_path), genotype_path,
          fingerprint_path, model_download_filepath, hidden_path,
          params['batch_size'], result_path, params['cuda_id'])

def candle_main():
    params = initialize_parameters()
    run(params)
    
if __name__ == "__main__":
    candle_main()
