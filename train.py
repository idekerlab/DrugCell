import candle
import os
from train_drugcell2 import main
import json
from json import JSONEncoder


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
    keys_parsing = ["train_data", "test_data", "val_data",
                    "onto", "genotype_hiddens", "fingerprint",
                    "genotype", "cell2id","drug2id", "drug_hiddens",
                    "model_name"]
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
    train_data_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['train_data']
    params['train_data'] = train_data_path
    test_data_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['test_data']
    params['test_data'] = test_data_path
    val_data_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['val_data']
    params['val_data'] = val_data_path
    onto_data_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['onto']
    params['ont'] = onto_data_path   
    cell2id_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['cell2id'] 
    params['cell2id'] = cell2id_path
    drug2id_path  = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['drug2id']
    params['drug2id'] = drug2id_path
    gene2id_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['gene2id']
    params['gene2id'] = gene2id_path
    genotype_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['genotype']
    params['genotype'] = genotype_path
    fingerprint_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['fingerprint']
    params['fingerprint'] = fingerprint_path
    hidden_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['hidden']
    params['hidden_path'] = hidden_path
#    output_dir_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['output_dir']
#    params['output_dir'] = output_dir_path
    result_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/" + params['result']
    params['result'] = result_path
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
#    print(params)
#    with open (json_out, 'w') as fp:
#        json.dump(params, fp, indent=4, cls=CustomEncoder)
    scores = main(params)
#    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
#        json.dump(scores, f, ensure_ascii=False, indent=4)
#    print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))


def candle_main():
    params = initialize_parameters()
    params =  preprocess(params)
    run(params)

if __name__ == "__main__":
    candle_main()
