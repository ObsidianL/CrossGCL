#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import dill
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utility import graph_to_sp,cal_ddi_loss, cal_bpr_loss, get_set_mul_hot, get_set2item_agg_graph
from util.metrics import metrics


class RepresentModel(nn.Module):
    def __init__(self, conf, graph_list):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]
        self.embedding_size = conf["embedding_size"]
        self.num_layers = conf["num_layer"]
        self.c_temp = conf["c_temp"]
  
        self.drug_sympset_graph, self.drug_symp_graph, self.sympset_symp_graph, self.symp_drugset_graph, self.drugset_drug_graph = graph_list 
        
        self.drugs_feature = nn.Parameter(torch.FloatTensor(self.conf['num_drugs'], self.embedding_size)) #[self.num_users, self.embedding_size]
        nn.init.xavier_normal_(self.drugs_feature)
        self.drugsets_feature = nn.Parameter(torch.FloatTensor(self.conf['num_drugsets'], self.embedding_size))
        nn.init.xavier_normal_(self.drugsets_feature)
        self.symps_feature = nn.Parameter(torch.FloatTensor(self.conf['num_symps'], self.embedding_size))
        nn.init.xavier_normal_(self.symps_feature)
        self.sympsets_feature = nn.Parameter(torch.FloatTensor(self.conf['num_sympsets'], self.embedding_size))
        nn.init.xavier_normal_(self.sympsets_feature)

        
        self.item_level_graph = graph_to_sp(self.drug_symp_graph,self.device)
        self.set_level_symp_drugset_graph = graph_to_sp(self.symp_drugset_graph,self.device)
        self.set_level_drug_sympset_graph = graph_to_sp(self.drug_sympset_graph,self.device)
        self.sympset_agg_graph = get_set2item_agg_graph(self.sympset_symp_graph, self.device)
        self.drugset_agg_graph = get_set2item_agg_graph(self.drugset_drug_graph, self.device)

        self.sympset_mul_hot_train, self.drugset_mul_hot_train = get_set_mul_hot(self.conf['num_symps'], self.conf['num_drugs'], 'train', self.device)
        self.sympset_mul_hot_val, self.drugset_mul_hot_val = get_set_mul_hot(self.conf['num_symps'], self.conf['num_drugs'], 'val', self.device)
        self.sympset_mul_hot_test, self.drugset_mul_hot_test = get_set_mul_hot(self.conf['num_symps'], self.conf['num_drugs'], 'test', self.device)

        ddi_A = dill.load(open("datasets/mimic/ddi_A_final.pkl",'rb'))
        self.ddi_A = torch.Tensor(ddi_A).to(self.device)
        
        
    def one_propagate(self, graph,  A_feature, B_feature):

        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature



    def propagate(self):
        IL_drugs_feature, IL_symps_feature = self.one_propagate(self.item_level_graph, self.drugs_feature, self.symps_feature)
        
        IL_sympsets_feature = torch.matmul(self.sympset_agg_graph, IL_symps_feature)
        IL_drugsets_feature = torch.matmul(self.drugset_agg_graph, IL_drugs_feature)

        SL_drugs_feature, SL_sympsets_feature = self.one_propagate(self.set_level_drug_sympset_graph, self.drugs_feature, self.sympsets_feature)
        SL_symps_feature, SL_drugsets_feature = self.one_propagate(self.set_level_symp_drugset_graph, self.symps_feature, self.drugsets_feature)
 
        return [IL_drugs_feature, SL_drugs_feature], [IL_sympsets_feature, SL_sympsets_feature], [IL_symps_feature, SL_symps_feature], [IL_drugsets_feature, SL_drugsets_feature]


    
    def cal_c_loss(self, pos, aug):
        # IL_drugs_feature, SL_drugs_feature
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :] # IL_drugs_feature_pos
        aug = aug[:, 0, :] # SL_drugs_feature_pos

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        
        neg = torch.concat([pos, aug], dim = 0)
        
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size] 相同节点不同视图下的评分
        
        # TODO 
        ttl_score = torch.matmul(pos, aug.T)
        
        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]
        # ttl_score *= 0.1
        c_loss = - torch.mean(torch.log(pos_score)-torch.log(ttl_score))

        return c_loss
    

    def cal_loss(self, sympsets_embedding, drugs_embedding, drugsets_embedding, symps_embedding, drug2drug_score):
        # [bs, 1, emb_size]
        IL_sympsets_feature, SL_sympsets_feature = sympsets_embedding
        # [bs, 1+neg_num, emb_size]
        IL_drugs_feature, SL_drugs_feature = drugs_embedding
        # [bs, 1, emb_size]
        IL_drugsets_feature, SL_drugsets_feature = drugsets_embedding
        # [bs, 1+neg_num, emb_size]
        IL_symps_feature, SL_symps_feature = symps_embedding


        # bpr loss
        pred_sympset2drugs = torch.sum(IL_sympsets_feature * IL_drugs_feature, 2) + torch.sum(SL_sympsets_feature * SL_drugs_feature, 2) #
        pred_drugset2symps = torch.sum(IL_drugsets_feature * IL_symps_feature, 2) + torch.sum(SL_drugsets_feature * SL_symps_feature, 2)
        
        alpha = 0.5
        bpr_loss = alpha*cal_bpr_loss(pred_sympset2drugs) + (1-alpha)*cal_bpr_loss(pred_drugset2symps) 
        

        # contrastive loss
        drug_cross_view_cl = self.cal_c_loss(IL_drugs_feature, SL_drugs_feature)
        symp_cross_view_cl = self.cal_c_loss(IL_symps_feature, SL_symps_feature)
        drugset_cross_view_cl = self.cal_c_loss(IL_drugsets_feature, SL_drugsets_feature)
        sympset_cross_view_cl = self.cal_c_loss(IL_sympsets_feature, SL_sympsets_feature)

        c_losses = [drug_cross_view_cl, symp_cross_view_cl, drugset_cross_view_cl, sympset_cross_view_cl]
        c_loss = sum(c_losses) / len(c_losses)

        # ddi_loss
        ddi_loss = cal_ddi_loss(drug2drug_score,self.ddi_A)

        return bpr_loss, c_loss, ddi_loss, [bpr_loss,c_loss, ddi_loss]

    def neg_sample(self, sympset_id, drugs, drugset_id, symps, drugs_feature, sympsets_feature, symps_feature, drugsets_feature):
        SL_symps_feature = symps_feature[1]
        IL_symps_feature = symps_feature[0]
        SL_drugs_feature = drugs_feature[1]
        IL_drugs_feature = drugs_feature[0]


        # sympsets_embedding = [2,2048,2,64] 
        IL_sympsets_emb = sympsets_feature[0][sympset_id].reshape(self.conf['batch_size'],self.embedding_size)
        SL_sympsets_emb = sympsets_feature[1][sympset_id].reshape(self.conf['batch_size'],self.embedding_size)
        
        # symps_embedding = [i[symps] for i in symps_feature]
        IL_drugsets_emb  = drugsets_feature[0][drugset_id].reshape(self.conf['batch_size'],self.embedding_size)
        SL_drugsets_emb = drugsets_feature[1][drugset_id].reshape(self.conf['batch_size'],self.embedding_size)
     
        sympset_drugs_score = torch.matmul(IL_sympsets_emb, IL_drugs_feature.T) + torch.matmul(SL_sympsets_emb, SL_drugs_feature.T)
        drugset_symps_score = torch.matmul(IL_drugsets_emb, IL_symps_feature.T) + torch.matmul(SL_drugsets_emb, SL_symps_feature.T) 
       
        sympset_drugs_score_neg = torch.sigmoid(sympset_drugs_score) * drugs[:,1:]
        drugset_symps_score_neg = torch.sigmoid(drugset_symps_score) * symps[:,1:]
        
   
        neg_nums = self.conf['neg_num']
        neg_drugs = torch.multinomial(sympset_drugs_score_neg,num_samples = neg_nums)
        neg_symps = torch.multinomial(drugset_symps_score_neg, num_samples=neg_nums)
        
        drugs = torch.concat([drugs[:,0:1], neg_drugs],dim = 1)
        symps = torch.concat([symps[:,0:1], neg_symps], dim = 1)
        
        return drugs, symps, sympset_drugs_score, drugset_symps_score   
    
    def forward(self, batch):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
       
        sympset_id,drugs,drugset_id,symps = batch 
        
        drugs_feature, sympsets_feature, symps_feature, drugsets_feature = self.propagate() 
        
        # drugs_embedding = [2,2048,2,64] 
        drugs_sample, symps_sample, sympset_drugs_score, drugset_symps_score = self.neg_sample(sympset_id, drugs, drugset_id, symps, drugs_feature, sympsets_feature, symps_feature, drugsets_feature)
        drugs_embedding = [i[drugs_sample] for i in drugs_feature]
        # sympsets_embedding = [2,2048,2,64] 
        sympsets_embedding = [i[sympset_id].expand(-1, drugs_sample.shape[1], -1) for i in sympsets_feature]
        
        symps_embedding = [i[symps_sample] for i in symps_feature]
        drugsets_embedding = [i[drugset_id].expand(-1, symps_sample.shape[1],-1) for i in drugsets_feature]
           
        ratio = 0.5
        agg_drug_feature = ratio*drugs_feature[0] + (1-ratio)*drugs_feature[1]
        drug2drug_score = torch.matmul(agg_drug_feature, agg_drug_feature.T)
        
        bpr_loss, c_loss, ddi_loss, loss_list = self.cal_loss(sympsets_embedding, drugs_embedding, drugsets_embedding,symps_embedding, drug2drug_score)

        return bpr_loss, c_loss, ddi_loss, loss_list

        

    
    def get_score(self, mode):
        drugs_feature, sympsets_feature, symps_feature, drugsets_feature = self.propagate()
        
        SL_symps_feature = symps_feature[1]
        IL_symps_feature = symps_feature[0]
        SL_drugs_feature = drugs_feature[1]
        IL_drugs_feature = drugs_feature[0]
        if mode == "train":
            sympset_emb_IL = self.sympset_mul_hot_train.matmul(IL_symps_feature)/self.sympset_mul_hot_train.sum(axis = 1).reshape([-1,1])
            sympset_emb_SL = self.sympset_mul_hot_train.matmul(SL_symps_feature)/self.sympset_mul_hot_train.sum(axis = 1).reshape([-1,1])
           
        if mode == 'val':
            sympset_emb_IL = self.sympset_mul_hot_val.matmul(IL_symps_feature)/self.sympset_mul_hot_val.sum(axis = 1).reshape([-1,1])
            sympset_emb_SL = self.sympset_mul_hot_val.matmul(SL_symps_feature)/self.sympset_mul_hot_val.sum(axis = 1).reshape([-1,1])
        if mode == 'test':
            sympset_emb_IL = self.sympset_mul_hot_test.matmul(IL_symps_feature)/self.sympset_mul_hot_test.sum(axis = 1).reshape([-1,1])
            sympset_emb_SL = self.sympset_mul_hot_test.matmul(SL_symps_feature)/self.sympset_mul_hot_test.sum(axis = 1).reshape([-1,1])
            
            
        alpha = self.conf['alpha']
        scores_II = sympset_emb_IL.matmul(IL_drugs_feature.T) 
        scores_SS = sympset_emb_SL.matmul(SL_drugs_feature.T)
        scores_IS = sympset_emb_IL.matmul(SL_drugs_feature.T)
        scores_SI = sympset_emb_SL.matmul(IL_drugs_feature.T)
        scores_IIIS = alpha * scores_II + (1-alpha)*scores_IS
            
        return scores_IIIS
        

    def get_score_feature(self, mode):
        drugs_feature, sympsets_feature, symps_feature, drugsets_feature = self.propagate()
        
        SL_symps_feature = symps_feature[1]
        IL_symps_feature = symps_feature[0]
        SL_drugs_feature = drugs_feature[1]
        IL_drugs_feature = drugs_feature[0]
        if mode == "train":
            sympset_emb_IL = self.sympset_mul_hot_train.matmul(IL_symps_feature)/self.sympset_mul_hot_train.sum(axis = 1).reshape([-1,1])
            sympset_emb_SL = self.sympset_mul_hot_train.matmul(SL_symps_feature)/self.sympset_mul_hot_train.sum(axis = 1).reshape([-1,1])
           
        if mode == 'val':
            sympset_emb_IL = self.sympset_mul_hot_val.matmul(IL_symps_feature)/self.sympset_mul_hot_val.sum(axis = 1).reshape([-1,1])
            sympset_emb_SL = self.sympset_mul_hot_val.matmul(SL_symps_feature)/self.sympset_mul_hot_val.sum(axis = 1).reshape([-1,1])
        
        if mode == 'test':
            sympset_emb_IL = self.sympset_mul_hot_test.matmul(IL_symps_feature)/self.sympset_mul_hot_test.sum(axis = 1).reshape([-1,1])
            sympset_emb_SL = self.sympset_mul_hot_test.matmul(SL_symps_feature)/self.sympset_mul_hot_test.sum(axis = 1).reshape([-1,1])

        scores_II = sympset_emb_IL.matmul(IL_drugs_feature.T) 
        scores_SS = sympset_emb_SL.matmul(SL_drugs_feature.T)
        scores_IS = sympset_emb_IL.matmul(SL_drugs_feature.T)
        scores_SI = sympset_emb_SL.matmul(IL_drugs_feature.T)
        return torch.concat([scores_II, scores_IS], dim = 1)


    def eval_with_fixed_threshold(self, thd, mode = "val"):
        scores = self.get_score(mode)
        y_pred = scores > thd
        if mode == 'val':
            y_gt = self.drugset_mul_hot_val
        if mode == 'test':
            y_gt = self.drugset_mul_hot_test 
        jac, f1, avgdrug, ddi = metrics(self.conf['num_drugs'], self.conf['num_symps']).get_metrics(y_gt,y_pred,self.ddi_A)
        return float(jac), float(f1), float(avgdrug), float(ddi)
        
        