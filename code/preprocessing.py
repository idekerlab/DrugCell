bac#!/usr/bin/env python
# coding: utf-8

# In[49]:


import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from drugcell_NN import *
import argparse
import numpy as np
import candle
import time


# In[73]:


CUDA_ID = 0
train = "../data/drugcell_train.txt"
test = "../data/drugcell_test.txt"
val = "../data/drugcell_val.txt"
onto = "../data/drugcell_ont.txt"
lr = 0.001
batchsize = 5000
gene2id = "../data/gene2ind.txt"
drug2id = "../data/drug2ind.txt"
cell2id = '../data/cell2ind.txt'
genotype_hiddens = 6
drug_hiddens='100,50,6'
final_hiddens=6
genotype="../data/cell2mutation.txt"
fingerprint='../data/drug2fingerprint.txt'
modeldir = "../MODEL"
epoch=5
num_hiddens_genotype = genotype_hiddens
num_hiddens_drug = list(map(int, drug_hiddens.split(',')))
num_hiddens_final = final_hiddens


# In[5]:


torch.set_printoptions(precision=5)


# In[6]:


def load_mapping(mapping_file):
    mapping = {}
    file_handle = open(mapping_file)
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    return mapping


# In[21]:


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


# In[22]:


gene2id_mapping = load_mapping(gene2id)
gene2id_mapping


# In[23]:


cell_features = np.genfromtxt(genotype, delimiter=',')
drug_features = np.genfromtxt(fingerprint, delimiter=',')
drug_features


# In[24]:


train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(train, test, cell2id, drug2id)


# In[26]:


num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])


# In[84]:


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


# In[85]:


dG, root, term_size_map, term_direct_gene_map = load_ontology(onto, gene2id_mapping)
root


# In[75]:


def create_term_mask(term_direct_gene_map, gene_dim):

    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))
        term_mask_map[term] = mask_gpu

    return term_mask_map


# In[76]:


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


# In[81]:


train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim,
            modeldir, epoch, batchsize, lr, num_hiddens_genotype, num_hiddens_drug, 
            num_hiddens_final, cell_features, drug_features)


# In[ ]:




