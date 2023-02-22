#!/usr/bin/env python3
"""Train DrugCell predictor."""
import logging
import sys
from time import time
import numpy as np
import torch
import candle
import pandas as pd
import sklearn
import os
import numpy as np
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import mean_absolute_error
from scipy.stats import spearmanr
#import code.utils.util
from code.utils.util import *
from code.drugcell_NN import *
import argparse
import numpy as np
import time
from time import time
#import sklearn
#from sklearn.metrics import r2_score, mean_absolute_error
#from scipy.stats import pearsonr, spearmanr


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



class Timer:
    """
    Measure runtime.
    """
    def __init__(self):
        self.start = time()

    def timer_end(self):
        self.end = time()
        time_diff = self.end - self.start
        return time_diff

    def display_timer(self, print_fn=print):
        time_diff = self.timer_end()
        if (time_diff)//3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
        else:
            print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )

def calc_mae(y_true, y_pred):
    return sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

def calc_r2(y_true, y_pred):
    target_mean = torch.mean(y_pred)
    ss_tot = torch.sum((y_pred - target_mean) ** 2)
    ss_res = torch.sum((y_pred - y_true) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def calc_pcc(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)
    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))


def create_term_mask(term_direct_gene_map, gene_dim, cuda):
    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
            mask_gpu = torch.autograd.Variable(mask.cuda(cuda))
            term_mask_map[term] = mask_gpu
    return term_mask_map

def main(params):
    train_data = params['train_data']
    test_data = params['test_data']
    val_data = params['val_data']
    genotype = params['genotype']
    fingerprint = params['fingerprint']
    onto = params['onto']
    cell2id = params['cell2id']
    drug2id = params['drug2id']
    gene2id = params['gene2id']
    output_dir = params['output_dir']
    model_name = params['model_name']
    hidden = params['hidden']
    result =  params['result']
    genotype_hiddens = params['genotype_hiddens']
    final_hiddens =  params['final_hiddens']
    cuda = params['cuda_id']
    drug_hiddens = params['drug_hiddens']
    
    logger = logging.getLogger(f'{model_name}')

    # Create model directory and dump files
    #model_dir = os.path.join(output_dir, model_name)
    model_dir = output_dir
    model_save_folder = output_dir
    os.makedirs(os.path.join(model_dir, 'hidden'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    #Load and parse inputs
    train_data, feature_dict, cell2id_mapping, drug2id_mapping = prepare_train_data(train_data, test_data,
                                                                                    cell2id, drug2id)
    gene2id_mapping = load_mapping(gene2id)

    # load cell/drug features
    cell_features = np.genfromtxt(genotype, delimiter=',')
    drug_features = np.genfromtxt(fingerprint, delimiter=',')
    num_cells = len(cell2id_mapping)
    num_drugs = len(drug2id_mapping)
    num_genes = len(gene2id_mapping)
    drug_dim = len(drug_features[0,:])
    
    # load ontology
    dG, root, term_size_map, term_direct_gene_map = load_ontology(onto, gene2id_mapping)
    
    # load the number of hiddens #######
    num_hiddens_genotype = genotype_hiddens
    num_hiddens_drug = list(map(int, drug_hiddens.split(',')))
    num_hiddens_final = final_hiddens
    #####################################
    CUDA_ID = cuda
    
    logger.info('train data has {0}'.format(len(train_data)))

    ### Params has been established
    timer = Timer()
    t = time()
    epoch_start_time = time()
    best_model = 0
    max_corr = 0

    save_top_model = os.path.join(model_dir, 'results/drugcell_{}_{}.pt')
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, num_genes,
                        drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final)
    train_feature, train_label, test_feature, test_label = train_data

    device = torch.device("cuda")
    model.to(device)
    model.cuda(CUDA_ID)
    
    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))
    term_mask_map = create_term_mask(model.term_direct_gene_map, num_genes, CUDA_ID)

    # Define optimizer
    optimizer_dict = {"adam": "optim.adam"}
    optim_value = params['optimizer']
    optimizer = optim.Adam(model.parameters(),
                           lr=params['learning_rate'],
                           betas=(0.9, 0.99),
                           eps=params['eps'])
    optimizer.zero_grad()
    timer.display_timer()

    ckpt = candle.CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)

    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)
    
    scores = {}
    epoch_list = []
    train_loss_list = []
    train_corr_list = []
    train_scc_list = []
    test_loss_list = []
    test_corr_list = []
    test_scc_list = []
    for name, param in model.named_parameters():
        term_name = name.split('_')[0]
        if '_direct_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1

    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label),
                                 batch_size=params['batch_size'], shuffle=False)

    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label),
                                batch_size=params['batch_size'], shuffle=False)

    for epoch in range(params['epochs']):
        model.train()
        epoch_list.append(epoch)
        train_predict =  torch.zeros(0,0).cuda(CUDA_ID)
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss_mean = 0
        t = time()    
        for i, (inputdata, labels) in enumerate(train_loader):
            features = build_input_vector(inputdata, cell_features, drug_features)
            cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

            optimizer.zero_grad()
            aux_out_map, _ = model(cuda_features)

            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['final'].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

            train_loss = 0
            count = 0
            for name, output in aux_out_map.items():
                count +=1
                loss = nn.MSELoss()
                if name == 'final':
                    train_loss += loss(output, cuda_labels)
                else:
                    train_loss += 0.2 * loss(output, cuda_labels)
            train_loss.backward()
#            train_loss_mean = train_loss/count
            train_loss_mean = train_loss
            for name, param in model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()
        #epoch_train_test_df['train_loss'] = train_loss_mean.cpu().detach().numpy()/len(train_loader)
        train_loss_list.append(train_loss_mean.cpu().detach().numpy()/len(train_loader))
        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss_mean / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )

        train_corr = pearson_corr(train_predict, train_label_gpu)
        train_corr_list.append(train_corr.cpu().detach().numpy())
        torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')
        train_predictions = np.array([p.cpu() for preds in train_predict for p in preds],dtype = np.float)
        train_predictions = train_predictions[0:len(train_predictions)]
        train_labels = np.array([l.cpu() for label in train_label_gpu for l in label],dtype = np.float)
        train_scc = spearmanr(train_labels, train_predictions)[0]
        train_scc_list.append(train_scc)
        model.eval()

        test_predict = torch.zeros(0,0).cuda(CUDA_ID)

        test_loss = 0
        tissue = []
        drug = []
        for i, (inputdata, labels) in enumerate(test_loader):
            features = build_input_vector(inputdata, cell_features, drug_features)
            cuda_features = Variable(features.cuda(CUDA_ID))
            aux_out_map, _ = model(cuda_features)
            values = inputdata.cpu().detach().numpy().tolist()
            keys = [i for i in feature_dict for x in values if feature_dict [i]== x ]
            tissue = [i.split(';')[0] for i in keys]
            drug = [i.split(';')[1] for i in keys]
            loss = nn.MSELoss()
            if test_predict.size()[0] == 0:
                test_predict = aux_out_map['final'].data
                loss_a =  loss(test_predict, cuda_labels)
                test_loss += loss_a.item() 
            else:
                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
                loss_a =  loss(test_predict, cuda_labels)
                test_loss += loss_a.item()
        logger.info(
            "\t **** TEST ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss / len(test_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        predictions = np.array([p.cpu() for preds in test_predict for p in preds] ,dtype = np.float )
        predictions = predictions[0:len(predictions)]
        labels = np.array([l.cpu() for label in labels for l in label],dtype = np.float)
        labels = labels[0:len(labels)]
        test_pearson_a = calc_pcc(torch.Tensor(predictions), torch.Tensor(labels))
        test_spearman_a = spearmanr(labels, predictions)[0]
        test_mean_absolute_error = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
        test_r2 = sklearn.metrics.r2_score(y_true=labels, y_pred=predictions)
        test_rmse_a = np.sqrt(np.mean((predictions - labels)**2))
        test_loss_a = test_loss / len(test_loader)
        epoch_end_time = time()
        test_loss_a = test_loss/len(test_loader)
#        test_loss_a = test_loss.cpu().detach().numpy()/len(test_loader)
        test_loss_list.append(test_loss_a)
        test_corr_list.append(test_pearson_a.cpu().detach().numpy())
        test_scc_list.append(test_spearman_a)
        if epoch == 0:
            min_test_loss = test_loss_a
            scores['test_loss'] = min_test_loss
            scores['test_pcc'] = test_pearson_a.cpu().detach().numpy().tolist()
            scores['test_MSE'] = test_mean_absolute_error
            scores['test_r2'] = test_r2
            scores['test_scc'] = test_spearman_a
        if test_loss_a < min_test_loss:
            min_test_loss = test_loss_a
            scores['test_loss'] = min_test_loss
            scores['test_pcc'] = test_pearson_a.cpu().detach().numpy().tolist()
            scores['test_MSE'] = test_mean_absolute_error
            scores['test_r2'] = test_r2
            scores['test_scc'] = test_spearman_a

        if test_pearson_a >= max_corr:
            max_corr = test_pearson_a
            best_model = epoch
            pred = pd.DataFrame({"Tissue": tissue, "Drug": drug, "True": labels, "Pred": predictions}).reset_index()
            pred_fname = str(model_dir+'/results/test_pred.csv')
            pred.to_csv(pred_fname, index=False)
#        print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\ttrain_loss\t%.6f\telapsed_time\t%s" % (epoch,
#                                                                                                                CUDA_ID,
#                                                                                                                train_corr, test_corr,
#                                                                                                                train_loss, epoch_end_time-epoch_start_time))
        epoch_start_time = epoch_end_time
        ckpt.ckpt_epoch(epoch, test_loss_a)
    torch.save(model, model_save_folder + '/model_final.pt')    
    print("Best performed model (epoch)\t%d" % best_model)
#    torch.save(save_top_model.format('epoch', '0', best_model))
    cols = ['epoch', 'train_loss', 'train_corr', 'test_loss', 'test_corr', 'test_scc_list']
    epoch_train_test_df = pd.DataFrame(columns=cols, index=range(params['epochs']))
    epoch_train_test_df['epoch'] = epoch_list
    epoch_train_test_df['train_loss'] = train_loss_list
    epoch_train_test_df['train_corr'] = train_corr_list
    epoch_train_test_df['train_scc'] = train_scc_list    
    epoch_train_test_df['test_loss'] = test_loss_list
    epoch_train_test_df['test_corr'] = test_corr_list
    epoch_train_test_df['test_scc'] = test_scc_list
    loss_results_name = str(model_dir+'/results/loss_results.csv')
    epoch_train_test_df.to_csv(loss_results_name, index=False)
    print(scores)
    return scores
    
if __name__ == "__main__":
    main()
