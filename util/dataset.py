#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import dill
import numpy as np
import scipy.sparse as sp 
from torch.utils.data import Dataset, DataLoader
from util.utility import pair_to_graph


class TrainDataset_DNS(Dataset):
    def __init__(self, sympset_drugset_pairs, num_drugs, num_symps, neg_sample=1):
        self.num_drugs = num_drugs
        self.num_symps = num_symps
        self.neg_sample = neg_sample

        self.sympset_drugset_pairs = []
        all_drugs = np.ones([self.num_drugs])
        all_symps = np.ones([self.num_symps])
        for sympset_id, sympset, drugset_id, drugset in sympset_drugset_pairs:
            sympset_mul = all_symps.copy()
            sympset_mul[sympset] = 0
            drugset_mul = all_drugs.copy()
            drugset_mul[drugset] = 0
            self.sympset_drugset_pairs.append([sympset_id, sympset, drugset_id, drugset,sympset_mul, drugset_mul])
            


    def __getitem__(self, index):
        sympset_id, sympset, drugset_id, drugset, sympset_mul, drugset_mul = self.sympset_drugset_pairs[index]
   
        pos_drug = np.random.choice(drugset)
        pos_symp = np.random.choice(sympset)
        drugset_mul = np.insert(drugset_mul,0,pos_drug)
        sympset_mul = np.insert(sympset_mul, 0, pos_symp)
                                                                                                               
        return torch.LongTensor([sympset_id]), torch.LongTensor(drugset_mul),torch.LongTensor([drugset_id]), torch.LongTensor(sympset_mul) 

    def __len__(self):
        return len(self.sympset_drugset_pairs)
    
    
class TestDataset_DNS(Dataset):
    def __init__(self, sympset_drugset_pairs, num_drugs, num_symps):
        self.num_drugs = num_drugs
        self.num_symps = num_symps

        self.sympset_drugset_pairs = []
        all_drugs = np.ones([self.num_drugs])
        all_symps = np.ones([self.num_symps])
        for sympset_id, sympset, drugset_id, drugset in sympset_drugset_pairs:
            sympset_mul = all_symps.copy()
            sympset_mul[sympset] = 0
            
            drugset_mul = all_drugs.copy()
            drugset_mul[drugset] = 0
            
            self.sympset_drugset_pairs.append([sympset_id, sympset, drugset_id, drugset,sympset_mul, drugset_mul])
            


    def __getitem__(self, index):
        sympset_id, sympset, drugset_id, drugset, sympset_mul, drugset_mul = self.sympset_drugset_pairs[index]
 
        pos_drug = np.random.choice(drugset)
        pos_symp = np.random.choice(sympset)
        drugset_mul = np.insert(drugset_mul,0,pos_drug)
        sympset_mul = np.insert(sympset_mul, 0, pos_symp)
                                                                                                               
        return torch.LongTensor([sympset_id]), torch.LongTensor(drugset_mul),torch.LongTensor([drugset_id]), torch.LongTensor(sympset_mul) 

    def __len__(self):
        return len(self.sympset_drugset_pairs)
    

class Datasets():
    def __init__(self, conf):
        self.conf = conf
        self.path = './datasets'
        self.name = conf['dataset']

        self.num_drugs, self.num_symps, self.num_drugsets, self.num_sympsets = self.get_data_size()

        self.graphs = self.get_graphs()
        

        sympset_drugset_pairs_train = self.get_sympset_drugset_pairs(task = "train")
        sympset_drugset_pairs_val = self.get_sympset_drugset_pairs(task = 'val')
        sympset_drugset_pairs_test = self.get_sympset_drugset_pairs(task = 'test')

        self.set_train_data = TrainDataset_DNS(sympset_drugset_pairs_train, self.num_drugs, self.num_symps , conf["neg_num"])
        self.set_val_data = TestDataset_DNS(sympset_drugset_pairs_val,  self.num_drugs, self.num_symps)
        self.set_test_data = TestDataset_DNS(sympset_drugset_pairs_test,   self.num_drugs, self.num_symps)

        self.train_loader = DataLoader(self.set_train_data, batch_size=self.conf['batch_size'], shuffle=True, num_workers=8, drop_last=True)
        self.val_loader = DataLoader(self.set_val_data, batch_size=self.conf['batch_size'], shuffle=False, num_workers=8)
        self.test_loader = DataLoader(self.set_test_data, batch_size=self.conf['batch_size'], shuffle=False, num_workers=8)

    def get_graphs(self):
        
        sympset_symp_graph = pair_to_graph('./datasets/mimic/sympset_symp_pair.pkl',shape=(self.num_sympsets, self.num_symps),is_weighted = False )
        drug_symp_graph = pair_to_graph('./datasets/mimic/drug_symp_pair_weight.pkl',shape =(self.num_drugs, self.num_symps),is_weighted = True )
        drugset_drug_graph = pair_to_graph('./datasets/mimic/drugset_drug_pair.pkl',shape=(self.num_drugsets, self.num_drugs),is_weighted = False )
        symp_drugset_graph = pair_to_graph('./datasets/mimic/symp_drugset_pair_weight.pkl',shape=(self.num_symps, self.num_drugsets),is_weighted = False )
        drug_sympset_graph = pair_to_graph('./datasets/mimic/drug_sympset_pair_weight.pkl',shape=(self.num_drugs, self.num_sympsets),is_weighted = False )
        
        return [drug_sympset_graph, drug_symp_graph, sympset_symp_graph,symp_drugset_graph,drugset_drug_graph]
        
    def get_data_size(self):
        data_size = dill.load(open("./datasets/mimic/mimic_data_size.pkl",'rb'))
        return data_size['drug'], data_size['symp'], data_size['drugset'],  data_size['sympset']



    def get_sympset_drugset_pairs(self, task = ""):
        with open('./datasets/mimic/sympset_drugset_{}_comp.pkl'.format(task), 'rb') as f:
            sympset_drugset_pairs = dill.load(f)
        return sympset_drugset_pairs
        

