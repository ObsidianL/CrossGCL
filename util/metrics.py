import torch
from util.utility import get_multi_label

class metrics():
    def __init__(self, num_drugs = 131, num_symps = 1958):
        self.num_drugs = num_drugs
        self.num_symps = num_symps
        
    def cal_jaccard(self, y_gt, y_pred):
        subsec = y_pred & y_gt
        union = y_pred | y_gt
        return (subsec.sum(axis = 1)/union.sum(axis = 1)).mean()
    
    def cal_f1(self,y_gt, y_pred):
        TP = (y_gt & y_pred).sum(dim = 1)
        precision = TP / y_gt.sum(dim = 1)
        recall = TP / y_pred.sum(dim = 1)
        f1 = 2*precision*recall/(precision + recall + 1e-9)
        return f1.mean()
    
    def cal_avgdrug_pred(self,y_pred):
        return y_pred.sum()/y_pred.shape[0]
    
    def cal_ddi(self,y_pred,ddi_A):
        all_cnt, ddi_cnt = 0, 0
        for idx in range(y_pred.shape[0]):
            preds = torch.where(y_pred[idx])[0]
            for i in range(len(preds)):
                for j in range(0, len(preds)):
                    if j <= i:
                        continue
                    if ddi_A[preds[i]][preds[j]] != 0 or ddi_A[preds[i]][preds[j]]:
                        ddi_cnt += 1
                    all_cnt += 1
        return ddi_cnt / all_cnt

    def get_metrics(self,y_gt, y_pred, ddi_A, is_multihot = True, is_ddi = False):
        if is_multihot == False:
            y_gt = get_multi_label(data = y_gt, shape = [len(y_gt), self.num_drugs])

        jac = self.cal_jaccard(y_gt, y_pred)
        f1 = self.cal_f1(y_gt, y_pred)
        avg_drug = self.cal_avgdrug_pred(y_pred)
        ddi = 0.0
        if is_ddi:
            ddi = self.cal_ddi(y_pred, ddi_A)
        
        return jac, f1, avg_drug, ddi
        
        
        
