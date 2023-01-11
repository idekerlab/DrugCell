import os
import candle
from util import *
from drugcell_NN import *

os.environ['CANDLE_DATA_DIR'] = '/research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/DrugCell/'
file_path = os.path.dirname(os.path.realpath(__file__))
print(file_path)

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
        "name": "load",
        "type": str, 
        "help": "pre build drugcell model",
    },
    {  
        "name": "result",
        "type": str, 
        "help": "result file name",
    },
]

# required definitions
required = [
    "predict",
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


def preprocess(params):
    test_file = params['test_data']
    train_file = params['train_data'] 
    cell2id_mapping_file = params['cell2id']
    drug2id_mapping_file = params['drug2id']
    

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)
    
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
    train_feature, train_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping

def run(params):
    keys_parsing = []

def candle_main():
    params = initialize_parameters()
    params = preprocess(params)
#    run(params)
    
if __name__ == "__main__":
    candle_main()
