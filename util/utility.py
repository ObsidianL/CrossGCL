#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp 
import torch
import dill

def get_multi_label(data,shape):
    data_mul = torch.zeros(size = shape)
    for i in range(len(data)):
        data_mul[i][data[i]] = 1
    return data_mul



def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)
        
    loss = - torch.log(torch.sigmoid((pos - negs))) # [bs]
    loss = torch.mean(loss)
    return loss

def cal_ddi_loss(drug2drug_score,ddi_A):
    ddi_score = (torch.sigmoid(drug2drug_score) * ddi_A).sum()
    return ddi_score
    
class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def get_set_mul_hot(num_symps, num_drugs, mode, device):
    data = dill.load(open(f'./datasets/mimic/sympset_drugset_{mode}.pkl','rb'))
    sympset = [x[0] for x in data]
    drugset_val = [x[1] for x in data]
    sympset_mul = np.zeros(shape = [len(sympset), num_symps])
    for i in range(len(sympset)):
        sympset_mul[i][sympset[i]] = True
    sympset_mul = torch.Tensor(sympset_mul).to(device)
    
    drugset_mul_hot = np.zeros(shape = [len(drugset_val), num_drugs])
    for i in range(len(drugset_val)):
        drugset_mul_hot[i][drugset_val[i]] = 1
    drugset_mul_hot = torch.BoolTensor(drugset_mul_hot).to(device)
    
    return sympset_mul, drugset_mul_hot

def graph_to_sp(graph, device):
    sp_graph = sp.bmat([[sp.csr_matrix((graph.shape[0], graph.shape[0])), graph], [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))]])    
    return to_tensor(laplace_transform(sp_graph)).to(device)
    

def get_set2item_agg_graph(set2item_graph,device):
    set_size = set2item_graph.sum(axis=1) + 1e-8
    set2item_graph = sp.diags(1/set_size.A.ravel()) @ set2item_graph
    return to_tensor(set2item_graph).to(device)

def pair_to_graph(file, shape, is_weighted):
    with open(file, 'rb') as f:
        pairs = dill.load(f)
    indice = np.array(pairs, dtype=np.int32)
    if is_weighted:
        values = indice[:,2]
    else:
        values = np.ones(len(pairs), dtype=np.float32)  
    graph = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=shape).tocsr()
    return graph

