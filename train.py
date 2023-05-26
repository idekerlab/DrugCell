import candle
import os
from train_drugcell2 import main
import json
from json import JSONEncoder
from preprocessing_new import mkdir

file_path = os.path.dirname(os.path.realpath(__file__))


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
        'DrugCell_params.txt',
        'pytorch',
        prog='DrugCell_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

def preprocess(params):
    keys_parsing = ["train_data", "test_data", "val_data", "onto_file",
                    "genotype_hiddens", "fingerprint",
                    "genotype", "cell2id","drug2id", "drug_hiddens",
                    "model_name"]
    print(os.environ['CANDLE_DATA_DIR'])
    print(os.environ['CANDLE_DATA_DIR'])
    #requirements go here
    data_dir = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Data/"
    model_params = {key: params[key] for key in keys_parsing}
    params['model_params'] = model_params
    args = candle.ArgumentStruct(**params)
    train_data_path = data_dir + params['train_data']
    params['train_data'] = train_data_path
    test_data_path = data_dir + params['test_data']
    params['test_data'] = test_data_path
    val_data_path = data_dir + params['val_data']
    params['val_data'] = val_data_path
    onto_data_path = data_dir + params['onto_file']
    params['onto'] = onto_data_path   
    cell2id_path = data_dir + params['cell2id'] 
    params['cell2id'] = cell2id_path
    drug2id_path  = data_dir + params['drug2id']
    params['drug2id'] = drug2id_path
    gene2id_path = data_dir + params['gene2id']
    params['gene2id'] = gene2id_path
    genotype_path = data_dir + params['genotype']
    params['genotype'] = genotype_path
    fingerprint_path = data_dir + params['fingerprint']
    params['fingerprint'] = fingerprint_path
    hidden_path = data_dir + params['hidden']
    params['hidden_path'] = hidden_path
    output_dir_path = data_dir + params['output']
    mkdir(output_dir_path)
    params['output_dir'] = output_dir_path
    params['result'] = data_dir + params['result']
    return(params)


class CustomData:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
            return o.__dict__


def run(params):
    params['data_type'] = str(params['data_type'])
    json_out = params['output_dir']+'/params.json'
    print(params)

    with open (json_out, 'w') as fp:
        json.dump(params, fp, indent=4, cls=CustomEncoder)

    scores = main(params)
    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
#    print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))


def candle_main():
    params = initialize_parameters()
    params =  preprocess(params)
    run(params)

if __name__ == "__main__":
    candle_main()
