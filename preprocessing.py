#!/homes/ac.rgnanaolivu/miniconda3/envs/rohan_python/bin/python

import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from code.drugcell_NN import *
import argparse
import numpy as np
import candle
import time
import logging
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

# setup logging

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-tr', dest='train_data', type=str, default='data/drugcell_train.txt',
        help='Path to the train drug sensitivity drugcell train data')
    parser.add_argument('-te', dest='test_data', type=str, default='data/drugcell_test.txt',
        help='Path to the test drug sensitivity drugcell data')
    parser.add_argument('-v', dest='val_data', type=str, default='data/drugcell_val.txt',
        help='Path to the drug sensitivity drugcell validation data')
    parser.add_argument('-ont', dest='ont_filepath', type=str,default='data/drugcell_ont.txt',
        help='Path to the drugcell ontology data')
    parser.add_argument('-geno', dest='genotype_path', type=str,default='data/cell2mutation.txt',
        help='Path to the genotype data')
    parser.add_argument('-CUDA', dest='CUDA_ID', type=int, default=1,
        help='value for CUDA ID, default =1')
    parser.add_argument('-f',dest='fingerprint_filepath', type=str, default='data/drug2fingerprint.txt',
        help='Path to the drug fingerprint file')
    parser.add_argument('-d', dest='drug2id', type=str,default='data/drug2ind.txt',
        help='Path to the drug2id file')    
    parser.add_argument('-g', dest='gene2id', type=str, default='data/gene2ind.txt',
        help='Path to the gene2id file')
    parser.add_argument('-c', dest='cell2id', type=str, default='data/cell2ind.txt',
        help='Path to the cell2id file')
    parser.add_argument('-o', dest='output_dir', 
        help='Directory where the model will be stored.')
    parser.add_argument('-l', dest='learning_rate', type=float,default=0.001,
        help='learning rate for the model, default 0.001')
    parser.add_argument('-b', dest='batchsize', type=int,default=5000,
        help='batch size for data processing, default 5000')
    parser.add_argument('-e', dest='epocs', type=int,default=300,
        help='total number of epochs')
    parser.add_argument('-dh', dest='drug_hiddens', type=str, default='100,50,6',
        help='total number of drug hiddens, default 100,50,6')
    parser.add_argument('-gh', dest='genotype_hiddens', type=int, default=6,
        help='total number of hidden genotypes')
    parser.add_argument('-fh', dest='final_hiddens', type=str, default=6,
        help='total number of final hiddens')
    args = parser.parse_args()
    return args

def load_mapping(mapping_file):
    mapping = {}
    file_handle = open(mapping_file)
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])
    file_handle.close()
    return mapping

def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            #print(tokens)
            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([float(tokens[2])])
    return feature, label

def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):
    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)
    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))
    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), 
            torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping

def load_ontology(file_name, gene2id_mapping):
    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}
    file_handle = open(file_name)
    gene_set = set()

    for line in file_handle:
        line = line.rstrip().split()
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        else:
            if line[1] not in gene2id_mapping:
                continue

            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[ line[0] ] = set()
            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])
            gene_set.add(line[1])

    file_handle.close()
    print('There are', len(gene_set), 'genes')

    for term in dG.nodes():
        term_gene_set = set()
        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]
        deslist = nxadag.descendants(dG, term)
        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term:', term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)
    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))
    print('There are', len(leaves), 'roots:', leaves[0])
    print('There are', len(dG.nodes()), 'terms')
    print('There are', len(connected_subG_list), 'connected componenets')

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print( 'There are more than connected components. Please connect them.')
        sys.exit(1)
    return dG, leaves[0], term_size_map, term_direct_gene_map

def create_term_mask(term_direct_gene_map, gene_dim):
    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))
        term_mask_map[term] = mask_gpu
    return term_mask_map

def train_model(root, term_size_map, term_direct_gene_map, dG, 
                train_data, gene_dim, drug_dim, model_save_folder, train_epochs, 
                batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, 
                num_hiddens_final, cell_features, drug_features):

    epoch_start_time = time.time()
    best_model = 0
    max_corr = 0

    # dcell neural network
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, 
                        num_hiddens_genotype, num_hiddens_drug, num_hiddens_final)

    train_feature, train_label, test_feature, test_label = train_data

    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))

    model.cuda(CUDA_ID)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

    optimizer.zero_grad()

    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1

    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

    for epoch in range(train_epochs):

        #Train
        model.train()
        train_predict = torch.zeros(0,0).cuda(CUDA_ID)

        for i, (inputdata, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)

            cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            # Here term_NN_out_map is a dictionary 
            aux_out_map, _ = model(cuda_features)

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['final'].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

            total_loss = 0
            for name, output in aux_out_map.items():
                loss = nn.MSELoss()
                if name == 'final':
                    total_loss += loss(output, cuda_labels)
                else: # change 0.2 to smaller one for big terms
                    total_loss += 0.2 * loss(output, cuda_labels)

                total_loss.backward()

            for name, param in model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                #print name, param.grad.data.size(), term_mask_map[term_name].size()
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()

        train_corr = pearson_corr(train_predict, train_label_gpu)

        #if epoch % 10 == 0:
        torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')

        #Test: random variables in training mode become static
        model.eval()

        test_predict = torch.zeros(0,0).cuda(CUDA_ID)

        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)
            cuda_features = Variable(features.cuda(CUDA_ID))

            aux_out_map, _ = model(cuda_features)

            if test_predict.size()[0] == 0:
                test_predict = aux_out_map['final'].data
            else:
                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

        test_corr = pearson_corr(test_predict, test_label_gpu)

        epoch_end_time = time.time()
        print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\ttotal_loss\t%.6f\telapsed_time\t%s" % (epoch, CUDA_ID, train_corr, test_corr, total_loss, epoch_end_time-epoch_start_time))
        epoch_start_time = epoch_end_time

        if test_corr >= max_corr:
            max_corr = test_corr
        best_model = epoch

    torch.save(model, model_save_folder + '/model_final.pt')	

    print("Best performed model (epoch)\t%d" % best_model)


def run(genotype_hiddens,  drug_hiddens, final_hiddens, gene2id,
        genotype, fingerprint, train,test, cell2id, drug2id, onto):
    torch.set_printoptions(precision=5)
    num_hiddens_genotype = genotype_hiddens
    num_hiddens_drug = list(map(int, drug_hiddens.split(',')))
    num_hiddens_final = final_hiddens
    gene2id_mapping = load_mapping(gene2id)
    cell_features = np.genfromtxt(genotype, delimiter=',')
    drug_features = np.genfromtxt(fingerprint, delimiter=',')
    train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(train, test, cell2id, drug2id)
    num_cells = len(cell2id_mapping)
    num_drugs = len(drug2id_mapping)
    num_genes = len(gene2id_mapping)
    drug_dim = len(drug_features[0,:])
    dG, root, term_size_map, term_direct_gene_map = load_ontology(onto, gene2id_mapping)
    print(root)
    #train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim,
    #modeldir, epoch, batchsize, lr, num_hiddens_genotype, num_hiddens_drug, 
    #            num_hiddens_final, cell_features, drug_features)
    
if __name__ == '__main__':
    # parse arguments
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    file_path = os.path.dirname(os.path.realpath(__file__))
    args = parse_args()
    run(args.genotype_hiddens,  args.drug_hiddens, args.final_hiddens, args.gene2id,
        args.genotype_path, args.fingerprint_filepath, args.train_data, args.test_data,
        args.cell2id, args.drug2id, args.ont_filepath)
    
