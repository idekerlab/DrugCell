import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.util import load_mapping
from utils.util import load_train_data
from utils.util import build_input_vector
from drugcell_NN import *
from utils.util import pearson_corr
import argparse
from pathlib import Path


def check_file(some_file):
    if os.path.isfile(some_file):
        print(some_file)
    else:
        print('{0} file does not exist'.format(some_file))
        exit()

def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
    torch_test_feature = torch.Tensor(test_feature)
    torch_test_label = torch.Tensor(test_label)
    torch_test_feature_label = (torch_test_feature, torch_test_label)
    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))
    return torch_test_feature_label, cell2id_mapping, drug2id_mapping
        

def main(predict, cell2id, drug2id, gene2id, genotype, fingerprint,
         load, hidden, batchsize, result, cuda):
    check_file(predict)
    check_file(cell2id)
    check_file(drug2id)
    predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(predict, cell2id, drug2id)
    gene2id_mapping = load_mapping(gene2id)

    # load cell/drug features
    cell_features = np.genfromtxt(genotype, delimiter=',')
    drug_features = np.genfromtxt(fingerprint, delimiter=',')
    
    num_cells = len(cell2id_mapping)
    num_drugs = len(drug2id_mapping)
    num_genes = len(gene2id_mapping)
    drug_dim = len(drug_features[0,:])
    
    CUDA_ID = cuda
    
    print("Total number of genes = %d" % num_genes)
    torch.set_printoptions(precision=5)
    predict_dcell(predict_data, num_genes, drug_dim, load, hidden, batchsize,
                  result, cell_features, drug_features, CUDA_ID)


def predict_dcell(predict_data, gene_dim, drug_dim, model_file, hidden_folder,
                  batch_size, result_file, cell_features, drug_features, CUDA_ID):

    feature_dim = gene_dim + drug_dim
    device = torch.device("cuda")
    
    model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)
    #model = torch.load(model_file, map_location='cuda:0')
    model.to(device)
    #model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)

    predict_feature, predict_label = predict_data

    predict_label_gpu = predict_label.cuda(CUDA_ID)
    model.cuda(CUDA_ID)
    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

    #Test
    test_predict = torch.zeros(0,0).cuda(CUDA_ID)
    term_hidden_map = {}

    batch_num = 0
    for i, (inputdata, labels) in enumerate(test_loader):
        # Convert torch tensor to Variable
        features = build_input_vector(inputdata, cell_features, drug_features)

        cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=False)

        # make prediction for test data
        aux_out_map, term_hidden_map = model(cuda_features)

        if test_predict.size()[0] == 0:
            test_predict = aux_out_map['final'].data
        else:
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

        for term, hidden_map in term_hidden_map.items():
            hidden_file = hidden_folder+'/'+term+'.hidden'
            with open(hidden_file, 'ab') as f:
                np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

        batch_num += 1

    test_corr = pearson_corr(test_predict, predict_label_gpu)
    print("Test pearson corr\t%s\t%.6f" % (model.root, test_corr))

    np.savetxt(result_file+'/drugcell.predict', test_predict.cpu().numpy(),'%.4e')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dcell')
    parser.add_argument('-p', dest='predict', help='Dataset to be predicted', required=True, type=Path)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
    parser.add_argument('-gene2id', help='Gene to ID mapping file', type=Path, default='data/gene2ind.txt')
    parser.add_argument('-drug2id', help='Drug to ID mapping file', type=Path, default='data/drug2ind.txt')
    parser.add_argument('-cell2id', help='Cell to ID mapping file', type=Path, default='data/cell2ind.txt')
    parser.add_argument('-load', help='Model file', type=str, default='Data/drugcell_v1.pt')
    parser.add_argument('-hidden', help='Hidden output folder', type=str, default='DrugCell/Hidden/')
    parser.add_argument('-result', help='Result file name', type=str, default='DrugCell/Result/')
    parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
    parser.add_argument('-genotype', help='Mutation information for cell lines', type=Path, required=True)
    parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=Path, required=True)    
    args = parser.parse_args()
    main(args.predict, args.cell2id, args.drug2id, args.gene2id,args.genotype, args.fingerprint, args.load,args.hidden, args.batchsize, args.result, args.cuda)
