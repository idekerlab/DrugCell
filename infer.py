import os
import candle
import pandas as pd
import torch
import torchvision
import numpy as np
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import mean_absolute_error
from scipy.stats import spearmanr
import torch.nn as nn
import torch.nn.functional as F
#from code.predict_drugcell import main
import sklearn
from code.utils.util import *
from code.drugcell_NN import *
from code.utils.util import load_mapping
from code.utils.util import load_train_data
from code.utils.util import build_input_vector
from code.utils.util import pearson_corr
from code.utils.util import prepare_predict_data
from time import time

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
        'DrugCell_params.txt',
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


def predict_dcell(predict_data, gene_dim, drug_dim, model_file, hidden_folder,
                  batch_size, result_file, cell_features, drug_features, CUDA_ID,output_dir):
    feature_dim = gene_dim + drug_dim
    device = torch.device("cuda")
    model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)
#    checkpoint = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)
    #model = torch.load(model_file, map_location='cuda:0')
    model.to(device)
#    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#   epoch = checkpoint['epoch']
#    loss = checkpoint['loss'] 
    #model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)

    predict_feature, predict_label, feature_dict = predict_data

    predict_label_gpu = predict_label.cuda(CUDA_ID)
    model.cuda(CUDA_ID)
    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)
    model_dir = output_dir

    #Test
    test_predict = torch.zeros(0,0).cuda(CUDA_ID)
    term_hidden_map = {}
    test_loss = 0
    batch_num = 0
    test_loss_list = []
    test_corr_list = []
    drug_list = []
    tissue_list = []
    print("Begin test evaluation")
    for i, (inputdata, labels) in enumerate(test_loader):
        # Convert torch tensor to Variable
        cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))
        features = build_input_vector(inputdata, cell_features, drug_features)
        cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=False)
        loss = nn.MSELoss()
        values = inputdata.cpu().detach().numpy().tolist()
        keys = [i for i in feature_dict for x in values if feature_dict [i]== x ]
        tissue = [i.split(';')[0] for i in keys]
        tissue_list.append(tissue)
        drug = [i.split(';')[1] for i in keys]
        drug_list.append(drug)
        # make prediction for test data
        aux_out_map, term_hidden_map = model(cuda_features)
        if test_predict.size()[0] == 0:
            test_predict = aux_out_map['final'].data
            loss_a =  loss(test_predict, cuda_labels)
            print(loss_a)
            test_loss += loss_a.item()
        else:
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
            loss_a =  loss(test_predict, cuda_labels)
            print(loss_a)
            test_loss += loss_a.item()
 #       for term, hidden_map in term_hidden_map.items():
 #           hidden_file = hidden_folder+'/'+term+'.hidden'
            #print(hidden_file)
 #           with open(hidden_file, 'ab') as f:
 #               np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

        batch_num += 1
    predictions = np.array([p.cpu() for preds in test_predict for p in preds] ,dtype = np.float )
    predictions = predictions[0:len(predictions)]
    labels = np.array([l.cpu() for label in labels for l in label],dtype = np.float)
    labels = labels[0:len(labels)]
    test_pearson_a = pearson_corr(torch.Tensor(predictions), torch.Tensor(labels))
    test_spearman_a = spearmanr(labels, predictions)[0]
    test_mean_absolute_error = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
    test_r2 = sklearn.metrics.r2_score(y_true=labels, y_pred=predictions)
    test_rmse_a = np.sqrt(np.mean((predictions - labels)**2))
    test_loss_a = test_loss / len(test_loader)
    epoch_end_time = time()
    test_loss_a = test_loss/len(test_loader)
    test_loss_list.append(test_loss_a)
    test_corr_list.append(test_pearson_a.cpu().detach().numpy())
    min_test_loss = test_loss_a
    print(test_loss_list)
    scores = {}
    scores['test_loss'] = min_test_loss
    scores['test_pcc'] = test_pearson_a.cpu().detach().numpy().tolist()
    scores['test_MSE'] = test_mean_absolute_error
    scores['test_r2'] = test_r2
    scores['test_scc'] = test_spearman_a
    test_corr = pearson_corr(test_predict, predict_label_gpu)
    print("Test pearson corr\t%s\t%.6f" % (model.root, test_corr))
    cols = ['drug', 'tissue', 'test_loss', 'test_corr']
    metrics_test_df = pd.DataFrame(columns=cols, index=range(len(test_loader)))
    metrics_test_df['test_loss'] = test_loss_list
    metrics_test_df['test_corr'] = test_corr_list
    loss_results_name = str(model_dir+'/results/test_metrics_results.csv')
    metrics_test_df.to_csv(loss_results_name, index=False)
    print(metrics_test_df)
    np.savetxt(result_file+'/drugcell.predict', test_predict.cpu().numpy(),'%.4e')

    
def run(params):
    keys_parsing = ["train_data", "test_data", "val_data",
                    "onto", "genotype_hiddens", "fingerprint",
                    "genotype", "cell2id","drug2id", "drug_hiddens",
                    "model_name"]
    model_param_key = []
    for key in params.keys():
        if key not in keys_parsing:
                model_param_key.append(key)
    model_params = {key: params[key] for key in model_param_key}
    params['model_params'] = model_params
    args = candle.ArgumentStruct(**params)
    cell2id_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['cell2id']
    drug2id_path  = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['drug2id']
    gene2id_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['gene2id']
    genotype_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['genotype']
    fingerprint_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['fingerprint']
    hidden_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['hidden']
    result_path = os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['result']
    val_data =  os.environ['CANDLE_DATA_DIR'] + "/DrugCell/Improve/Data/" + params['val_data']
    trained_model = params['data_model']
    hidden =  params['drug_hiddens']
    batchsize = params['batch_size']
    cell_features = np.genfromtxt(genotype_path, delimiter=',')
    drug_features = np.genfromtxt(fingerprint_path, delimiter=',')
    CUDA_ID = params['cuda_id']
    num_cells = len(cell2id_path)
    num_drugs = len(drug2id_path)
    num_genes = len(gene2id_path)
    drug_dim = len(drug_features[0,:])
    output_dir = params['output_dir']
    predict_data = prepare_predict_data(val_data, cell2id_path, drug2id_path)
    predict_dcell(predict_data, num_genes, drug_dim, trained_model, hidden_path, batchsize,
                  result_path, cell_features, drug_features, CUDA_ID, output_dir)


def candle_main():
    params = initialize_parameters()
    run(params)
    
if __name__ == "__main__":
    candle_main()
