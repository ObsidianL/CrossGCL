#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,sys
os.chdir(sys.path[0])

import argparse
from tqdm import tqdm
import torch
import torch.optim as optim

from util.utility import AutomaticWeightedLoss
from util.dataset import Datasets
from models.graph_represent_module import RepresentModel
from models.drug_size_module import DrugSizeModule
from util.metrics import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default='train', help="train or test mode")
parser.add_argument('--model', type=str, default='CrossGCL', help="model name")
parser.add_argument('--resume_path', type=str, default="", help='resume path for graph learning')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for graph learning')
parser.add_argument('--l2_reg', type=float, default=0.0001, help='l2 regularization for graph learning')
parser.add_argument('--c_lambda', type=float, default=0.1, help='contrastive loss weight for graph learning')
parser.add_argument('--ddi_lambda', type=float, default=0.00001, help='contrastive loss weight for graph learning')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs graph learning train')
parser.add_argument('--test_interval', type=int, default=5, help='Interval for eval of graph learning')
parser.add_argument('--neg_num', type=int, default=5, help='represent learnig bpr negative sample num for graph learning')
parser.add_argument('--num_layer', type=int, default=1, help='number of layer of the gcn')
parser.add_argument('--c_temp', type=float, default=0.25, help='temperature in the contrastive loss for graph learning')
parser.add_argument('--alpha', type=float, default=0.1, help='item view and set view drug score weight for  graph learning ')
parser.add_argument('--threshold', type=float, default=0.3, help='the threshold to eval the represent learning module')
parser.add_argument('--gamma', type=float, default=0.995, help='The decay coefficient of the learning rate exponential decay strategy for graph learning')
parser.add_argument('--cal_ddi', type=bool, default=False, help='Whether to calculate ddi during graph learning training')
parser.add_argument('--batch_size', type=int, default=2048, help='batch_size for graph learning')
parser.add_argument('--catboost_lr', type=float, default=0.01, help='learning rate for catboost')
parser.add_argument('--catboost_max_depth', type=int, default=7, help='max_depth for catboost')
parser.add_argument('--dataset', type=str, default='mimic', help='max_depth for catboost')
parser.add_argument('--embedding_size', type=int, default=64, help='')


parser.add_argument('--gpu', default="0", type=str, help="which gpu to use")

conf = parser.parse_args()
conf = vars(conf)

conf['model_save_path'] ='./checkpoints/{}/{}'.format(conf['dataset'],conf['model'])    


def gcn_represet_eval(model, thd, mode, is_ddi = False):
    scores = model.get_score(mode)
    y_pred = scores > thd
    if mode == 'val':
        y_gt = model.drugset_mul_hot_val
    if mode == 'test':
        y_gt = model.drugset_mul_hot_test 
    jac, f1, avgdrug, ddi = metrics(model.conf['num_drugs'], model.conf['num_symps']).get_metrics(y_gt,y_pred,model.ddi_A, is_multihot= True, is_ddi = is_ddi)
    return float(jac), float(f1), float(avgdrug), float(ddi)


def gcn_represet_train(model, conf, dataset, device):
    # awl = AutomaticWeightedLoss(6)
    # optimizer = optim.Adam([
    #             {'params': model.parameters(),'lr':conf['lr'], "weight_decay": conf["l2_reg"]},
    #             {'params': awl.parameters(),'weight_decay':0}])
    optimizer = optim.Adam([
                {'params': model.parameters(),'lr':conf['lr'], "weight_decay": conf["l2_reg"]}])
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=conf['gamma'])

    best_jac = 0
    for epoch in range(conf['epochs']):
        model.train(True)
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        #batch [userid, [pos_set, neg_set]] userid, pos_set, neg_set = [2048,1]
        for _, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]       
            bpr_loss, c_loss, ddi_loss, loss_list = model(batch)
            loss = bpr_loss + conf["c_lambda"] * c_loss + conf['ddi_lambda'] * ddi_loss
            # loss = awl(*loss_list)
            loss.backward()
            optimizer.step()
            
            loss_scalar = loss.detach()
            bpr_loss_scalar = bpr_loss.detach()
            c_loss_scalar = c_loss.detach()
            ddi_loss_scalar = ddi_loss.detach()

            pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f, ddi_loss: %.4f" 
                                 %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar, ddi_loss_scalar))
        
        model.eval()
        print("lr = ",lr_scheduler.get_last_lr())
        if (epoch + 1) % conf['test_interval'] == 0:
            if conf['cal_ddi'] == True:
                # jac, f1, avg_drug, ddi = model.eval_with_fixed_threshold(thd = conf['threshold'], mode = 'val', is_ddi = False)
                jac, f1, avg_drug, ddi = gcn_represet_eval(model, conf['threshold'], mode = 'val', is_ddi = True)
                
                print(f"valid: jac = {jac:.4f}, f1 = {f1:.4f}, avgdrug = {avg_drug:.4f}, ddi = {ddi:.4f}")
                
            else:
                # jac, f1, avg_drug, ddi = model.eval_with_fixed_threshold(thd = conf['threshold'], mode = 'val', is_ddi = True)
                jac, f1, avg_drug, ddi = gcn_represet_eval(model, conf['threshold'], mode = 'val', is_ddi = False)
                print(f"valid: jac = {jac:.4f}, f1 = {f1:.4f}, avgdrug = {avg_drug:.4f}")
                
            save_path = f"{conf['model_save_path']}/graph_rep_epoch={epoch}_jac={jac}.pth"
            torch.save(model.state_dict(), save_path)
            if jac > best_jac:
                best_jac = max(best_jac, jac)
                save_path = f"{conf['model_save_path']}/best_jac.pth"
                torch.save(model.state_dict(), save_path)
                best_jac = max(best_jac, jac)

        lr_scheduler.step()
    return model
        
        
def recommend_drugs_eval(drug_size_module, CatBR, mode = "test"):
    X_train, X_val, X_test = drug_size_module.get_feature()
    pred_len_val = CatBR.predict(X_val)
    pred_len_test = CatBR.predict(X_test)
    if mode == 'val':
        drugset = drug_size_module.drugset_val
        scores = drug_size_module.scores_val 
        pred_len = pred_len_val
        y_gt = drug_size_module.model.drugset_mul_hot_val
        
    if mode == 'test':
        drugset = drug_size_module.drugset_test
        scores = drug_size_module.scores_test
        pred_len = pred_len_test
        y_gt = drug_size_module.model.drugset_mul_hot_test
    

    thd = []
    for i in range(len(drugset)):
        ground = set(drugset[i])
        ll = min(max(12,int(round(pred_len[i],0))), 50)
        min_score = torch.topk(torch.Tensor(scores[i]),ll ).values[-1]
        thd.append(min_score)
    thd = torch.tensor(thd).reshape(-1,1)
    y_pred = torch.ge(scores, thd.to(drug_size_module.model.conf['device'])) 
    jac, f1, avgdrug, ddi = metrics(drug_size_module.model.conf['num_drugs'], drug_size_module.model.conf['num_symps']).get_metrics(y_gt,y_pred,drug_size_module.model.ddi_A, is_ddi = True)
    return float(jac), float(f1), float(avgdrug), float(ddi)
    


def main(conf):
    device = torch.device("cuda:{}".format(conf['gpu']) if torch.cuda.is_available() else "cpu")
    conf["device"] = "cuda:{}".format(conf['gpu']) if torch.cuda.is_available() else "cpu" 
    
    dataset = Datasets(conf)
    conf['num_drugs'] = dataset.num_drugs
    conf['num_sympsets'] = dataset.num_sympsets
    conf['num_symps'] = dataset.num_symps
    conf['num_drugsets'] = dataset.num_drugsets
    
    rep_model = RepresentModel(conf, dataset.graphs).to(device) #dataset.graphs = [u_b_graph_train, u_i_graph, b_i_graph]
    
    if not os.path.isdir(f"{conf['model_save_path']}"):
        os.makedirs(f"{conf['model_save_path']}")

    if conf['resume_path'] != "":
        PretrainedDict = torch.load(conf['resume_path'], map_location='cpu')
        rep_model.load_state_dict(PretrainedDict)
    


    if conf['mode'] == 'train':
        rep_model = gcn_represet_train(rep_model, conf, dataset, device)
        
    if conf['mode'] == 'infer':
        if conf['resume_path'] == "":
            print("please test with trained model")
        print("infer & test with trained model: {}".format(conf['resume_path']))
        rep_model.eval()
        drug_size_module = DrugSizeModule(rep_model)
        CatBR = drug_size_module.train(target_idx = 1, lr = conf['catboost_lr'], max_depth= conf['catboost_max_depth'])

        jac, f1, avgdrug, ddi = recommend_drugs_eval(drug_size_module, CatBR, mode = "test")
        print(f"test result: jac = {jac}, f1 = {f1}, avgdrug = {avgdrug}, ddi = {ddi}")
        

if __name__ == "__main__":
    main(conf)