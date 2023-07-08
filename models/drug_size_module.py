import numpy as np
import torch
import torch.nn as nn
import dill
from catboost import CatBoostRegressor


class DrugSizeModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        data_train = dill.load(open('./datasets/mimic/sympset_drugset_train.pkl','rb'))
        self.sympset_train = [x[0] for x in data_train]
        self.drugset_train = [x[1] for x in data_train]

        data_val = dill.load(open('./datasets/mimic/sympset_drugset_val.pkl','rb'))
        self.sympset_val = [x[0] for x in data_val]
        self.drugset_val = [x[1] for x in data_val]

        data_test = dill.load(open('./datasets/mimic/sympset_drugset_test.pkl','rb'))
        self.sympset_test = [x[0] for x in data_test]
        self.drugset_test = [x[1] for x in data_test]
        
        self.scores_train =model.get_score("train")
        self.scores_val = model.get_score("val")
        self.scores_test = model.get_score("test")
        
    def best_len(self, scores, drugset_mul_hot):
        jacs = []
        for i in range(1,scores.shape[-1]+1):
            top_i = torch.topk(input = scores, k = i, dim = -1)
            topi_mul = scores >= top_i.values[:,-1].reshape(-1,1)
            jac = (topi_mul & drugset_mul_hot).sum(dim = 1)/(topi_mul | drugset_mul_hot).sum(dim = 1)
            jacs.append(jac.reshape(-1,1))
        jacs = torch.concat(jacs, dim = 1)
        best_jacs = torch.topk(input = jacs, k = 1, dim = -1)
        best_lens = best_jacs.indices + 1
        ll = drugset_mul_hot.sum(dim = 1).reshape(-1,1)
        best_lens = torch.concat([best_lens, ll], dim= 1) 
        return best_lens
    

    def get_feature(self):
        x_train,x_val, x_test = [], [], []
        x_train.append(np.array(self.scores_train.cpu().detach()))
        x_val.append(np.array(self.scores_val.cpu().detach()))
        x_test.append(np.array(self.scores_test.cpu().detach()))
        
        x_train.append(self.model.sympset_mul_hot_train.cpu().detach().numpy().astype("int8"))
        x_val.append(self.model.sympset_mul_hot_val.cpu().detach().numpy().astype("int8"))
        x_test.append(self.model.sympset_mul_hot_test.cpu().detach().numpy().astype("int8"))
        
        x_train.append(np.array([[len(ele)] for ele in self.sympset_train]))
        x_val.append(np.array([[len(ele)] for ele in self.sympset_val]))
        x_test.append(np.array([[len(ele)] for ele in self.sympset_test]))
        
        x_train = np.concatenate(x_train,axis = 1)
        x_val = np.concatenate(x_val,axis = 1)
        x_test = np.concatenate(x_test,axis = 1)
        
        return x_train, x_val, x_test

        
    def train(self, target_idx = 0, lr = 0.01, max_depth = 3):
        X_train, X_val, X_test = self.get_feature()

        best_lens_train = self.best_len(self.scores_train, self.model.drugset_mul_hot_train)
        best_lens_val = self.best_len(self.scores_val, self.model.drugset_mul_hot_val)
        y_train = best_lens_train[:,target_idx].reshape(-1,1).cpu().numpy()
        y_val = best_lens_val[:, target_idx].reshape(-1,1).cpu().numpy()
        
        CatBR = CatBoostRegressor(iterations=100000, learning_rate=lr, loss_function='RMSE',eval_metric='RMSE',
                                  max_depth = max_depth,devices="cuda:0")
        CatBR.fit(X_train,y_train, eval_set=[(X_val,y_val)],verbose=100, early_stopping_rounds=1000)
        
        return CatBR


    



        
        
