import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import torch
import torch.nn as nn


def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))

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

        # jisoo
        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term:', term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)

    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
    #leaves = [n for n,d in dG.in_degree() if d==0]

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


def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []
    feature_dict = {}
    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            keys = list(cell2id.keys())[list(cell2id.values()).index(cell2id[tokens[0]])] + ";" + list(drug2id.keys())[list(drug2id.values()).index(drug2id[tokens[1]])]
            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            feature_dict[keys] = [cell2id[tokens[0]], drug2id[tokens[1]]]
            label.append([float(tokens[2])])
    return feature, label, feature_dict


def load_mapping(some_file):
    mapping = {}
    with  open(some_file) as fin:
        for line in fin:
            line = line.rstrip().split()
            mapping[line[1]] = int(line[0])
    return mapping

def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)
    test_feature, test_label, feature_dict = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
#    test_feature = test_feature_dict.values()
    torch_test_feature = torch.Tensor(test_feature)
    torch_test_label = torch.Tensor(test_label)
    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))
    return torch_test_feature, torch_test_label, feature_dict


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label, feature_dict  = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label, feature_dict  = load_train_data(test_file, cell2id_mapping, drug2id_mapping)
#    train_feature = list(train_feature_dict.values())
#    test_feature = list(test_feature_dict.values())

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))
#    return (train_feature_dict, train_label, test_feature_dict, test_label), cell2id_mapping, drug2id_mapping
    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), feature_dict, cell2id_mapping, drug2id_mapping


def build_input_vector(input_data, cell_features, drug_features):
    genedim = len(cell_features[0,:])
    drugdim = len(drug_features[0,:])
    feature = np.zeros((input_data.size()[0], (genedim+drugdim)))

    for i in range(input_data.size()[0]):
        feature[i] = np.concatenate((cell_features[int(input_data[i,0])], drug_features[int(input_data[i,1])]), axis=None)

    feature = torch.from_numpy(feature).float()
    return feature


def pearsonr(x, y):
    """Compute Pearson correlation.
    Args:
        x (torch.Tensor): 1D vector
        y (torch.Tensor): 1D vector of the same size as y.
    Raises:
        TypeError: not torch.Tensors.
        ValueError: not same shape or at least length 2.
    Returns:
        Pearson correlation coefficient.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Function expects torch Tensors.')

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(' x and y must be 1D Tensors.')

    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')

    if len(x) < 2:
        raise ValueError('x and y must have length at least 2.')

    # If an input is constant, the correlation coefficient is not defined.
    if bool((x == x[0]).all()) or bool((y == y[0]).all()):
        raise ValueError('Constant input, r is not defined.')

    mx = x - torch.mean(x)
    my = y - torch.mean(y)
    cost = (
        torch.sum(mx * my) /
        (torch.sqrt(torch.sum(mx**2)) * torch.sqrt(torch.sum(my**2)))
    )
    return torch.clamp(cost, min=-1.0, max=1.0)


def correlation_coefficient_loss(labels, predictions):
    """Compute loss based on Pearson correlation.
    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values
    Returns:
        torch.Tensor: A loss that when minimized forces high squared correlation coefficient:
        \$1 - r(labels, predictions)^2\$  # noqa
    """
    return 1 - pearsonr(labels, predictions)**2


def mse_cc_loss(labels, predictions):
    """Compute loss based on MSE and Pearson correlation.
    The main assumption is that MSE lies in [0,1] range, i.e.: range is
    comparable with Pearson correlation-based loss.
    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values
    Returns:
        torch.Tensor: A loss that computes the following:
        \$mse(labels, predictions) + 1 - r(labels, predictions)^2\$  # noqa
    """
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(predictions, labels)
    cc_loss = correlation_coefficient_loss(labels, predictions)
    return mse_loss + cc_loss
