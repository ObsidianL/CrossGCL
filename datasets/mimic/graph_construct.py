import numpy as np
import dill


def flat(records):
    data = []
    for user in records:
        for vis in user:
            data.append([vis[0], vis[2]])
    return data

def train_valid_test_split(data_file):
    data_full = dill.load(open(data_file,'rb'))
    split_point = int(len(data_full) * 2 / 3)
    data_train = data_full[:split_point]
    eval_len = int(len(data_full[split_point:]) / 2)
    data_val = data_full[split_point:split_point + eval_len]
    data_test = data_full[split_point+eval_len:]
    
    data_full,data_train,data_val, data_test = flat(data_full),flat(data_train),flat(data_val),flat(data_test)
    
    return data_full, data_train, data_val, data_test

   

def get_set_voc(data_full):
    drug_set = set()
    symp_set = set()
    symp_set_set = set()
    drug_set_set = set()
    for symptoms, drugs in data_full:
        drug_set = drug_set.union(drugs)
        symp_set = symp_set.union(symptoms)
        symp_set_set.add(tuple(sorted(symptoms)))
        drug_set_set.add(tuple(sorted(drugs)))



    sympset2id = dict()
    drugset2id = dict()
    id2sympset = dict()
    id2drugset = dict()
    for symset in symp_set_set:
        if symset not in sympset2id:
            sympset2id[symset] = len(sympset2id)
            id2sympset[sympset2id[symset]] = symset
    for drugset in drug_set_set:
        if drugset not in drugset2id:
            drugset2id[drugset] = len(drugset2id)
            id2drugset[drugset2id[drugset]] = drugset
    setvoc = dict()
    setvoc['sympset2id'] = sympset2id
    setvoc['drugset2id'] = drugset2id
    setvoc['id2sympset'] = id2sympset
    setvoc['id2drugset'] = id2drugset
    
    return setvoc, drug_set, symp_set, drug_set_set, symp_set_set


def get_data_record_comp(data,setvoc):
    data_comp = []
    for sympset, drugset in data:
        sympset_id = setvoc['sympset2id'][tuple(sorted(sympset))]
        drugset_id = setvoc['drugset2id'][tuple(sorted(drugset))]
        data_comp.append([sympset_id,sympset,drugset_id,drugset])
    return data_comp 


if __name__ == "__main__":

    data_full, data_train, data_val, data_test = train_valid_test_split('records_final.pkl')
    dill.dump(data_val, open('sympset_drugset_val.pkl','wb'))
    dill.dump(data_test, open('sympset_drugset_test.pkl','wb'))
    dill.dump(data_train, open('sympset_drugset_train.pkl','wb'))

    setvoc, drug_set, symp_set, drug_set_set, symp_set_set = get_set_voc(data_full)
    
    data_size = {"drug":len(drug_set),
                "symp":len(symp_set),
                "drugset":len(drug_set_set),
                "sympset":len(symp_set_set)}
    dill.dump(data_size, open("mimic_data_size.pkl",'wb'))


    data_train_comp, data_val_comp, data_test_comp = get_data_record_comp(data_train, setvoc), get_data_record_comp(data_val, setvoc), get_data_record_comp(data_test, setvoc)
    dill.dump(data_train_comp, open('sympset_drugset_train_comp.pkl','wb'))
    dill.dump(data_val_comp, open('sympset_drugset_val_comp.pkl','wb'))
    dill.dump(data_test_comp, open('sympset_drugset_test_comp.pkl','wb'))






    drug_symp_graph = np.zeros([len(drug_set), len(symp_set)])
    for symptoms, drugs in data_train:
        for drug in drugs:
            for symp in symptoms:
                drug_symp_graph[drug][symp] += 1  
    print(drug_symp_graph.shape)




    drug_symp_pair = []
    for i in range(len(drug_symp_graph)):
        for j in range(len(drug_symp_graph[0])):
            if drug_symp_graph[i][j] > 0.5:
                drug_symp_pair.append((i,j,drug_symp_graph[i][j]))
    dill.dump(drug_symp_pair, open('drug_symp_pair_weight.pkl','wb'))




    drug_sympset_graph = np.zeros([len(drug_set), len(symp_set_set)])
    for symptoms, drugs in data_train:
        for drug in drugs:
            sympset = tuple(sorted(symptoms))
            iid = setvoc['sympset2id'][sympset]
            drug_sympset_graph[drug][iid] += 1  
    print(drug_sympset_graph.shape)


    drug_sympset_pair = []
    for i in range(len(drug_sympset_graph)):
        for j in range(len(drug_sympset_graph[0])):
            if drug_sympset_graph[i][j] > 0.5:
                drug_sympset_pair.append((i,j,drug_sympset_graph[i][j]))
    dill.dump(drug_sympset_pair, open('drug_sympset_pair_weight.pkl','wb'))




    drug_sympset_graph_val = np.zeros([len(drug_set), len(symp_set_set)])
    for symptoms, drugs in data_val:
        for drug in drugs:
            sympset = tuple(sorted(symptoms))
            iid = setvoc['sympset2id'][sympset]
            drug_sympset_graph_val[drug][iid] += 1  
    print(drug_sympset_graph_val.shape)

    drug_sympset_pair_val = []
    for i in range(len(drug_sympset_graph_val)):
        for j in range(len(drug_sympset_graph_val[0])):
            if drug_sympset_graph_val[i][j] > 0.5:
                drug_sympset_pair_val.append((i,j,drug_sympset_graph_val[i][j]))
    dill.dump(drug_sympset_pair_val, open('drug_sympset_pair_weight_val.pkl','wb'))



    # drug-sympset graph&pairs
    drug_sympset_graph_test = np.zeros([len(drug_set), len(symp_set_set)])
    for symptoms, drugs in data_test:
        for drug in drugs:
            sympset = tuple(sorted(symptoms))
            iid = setvoc['sympset2id'][sympset]
            drug_sympset_graph_test[drug][iid] += 1  
    print(drug_sympset_graph_test.shape)


    drug_sympset_pair_test = []
    for i in range(len(drug_sympset_graph_test)):
        for j in range(len(drug_sympset_graph_test[0])):
            if drug_sympset_graph_test[i][j] > 0.5:
                drug_sympset_pair_test.append((i,j,drug_sympset_graph_test[i][j]))
    dill.dump(drug_sympset_pair_test, open('drug_sympset_pair_weight_test.pkl','wb'))



    # drugset-symp graph&pair
    drugset_symp_graph = np.zeros([len(drug_set_set), len(symp_set)])
    for symptoms, drugs in data_train:
        for symp in symptoms:
            drugset = tuple(sorted(drugs))
            iid = setvoc['drugset2id'][drugset]
            drugset_symp_graph[iid][symp] += 1  
    print(drugset_symp_graph.shape)




    drugset_symp_pair = []
    for i in range(len(drugset_symp_graph)):
        for j in range(len(drugset_symp_graph[0])):
            if drugset_symp_graph[i][j] > 0.5:
                drugset_symp_pair.append((i,j,drugset_symp_graph[i][j]))
    dill.dump(drugset_symp_pair, open('drugset_symp_pair_weight.pkl','wb'))



    #symp-drugset graph & pair
    symp_drugset_graph = np.zeros([len(symp_set), len(drug_set_set)])
    for symptoms, drugs in data_train:
        for symp in symptoms:
            drugset = tuple(sorted(drugs))
            iid = setvoc['drugset2id'][drugset]
            symp_drugset_graph[symp][iid] += 1  
    print(symp_drugset_graph.shape)


    symp_drugset_pair = []
    for i in range(len(symp_drugset_graph)):
        for j in range(len(symp_drugset_graph[0])):
            if symp_drugset_graph[i][j] > 0.5:
                symp_drugset_pair.append((i,j,symp_drugset_graph[i][j]))
    dill.dump(symp_drugset_pair, open('symp_drugset_pair_weight.pkl','wb'))




    # drugset-drug graph
    drugset_drug_graph = np.zeros([len(drug_set_set), len(drug_set)])
    drugset_drug_pair = []
    for drugsetid, drugset in setvoc['id2drugset'].items():
        for drug in drugset:
            drugset_drug_graph[drugsetid][drug] = 1
            drugset_drug_pair.append((drugsetid,drug,1)) 
    print(drugset_drug_graph.shape)
    dill.dump(drugset_drug_graph, open('drugset_drug_graph.pkl','wb'))
    dill.dump(drugset_drug_pair, open('drugset_drug_pair.pkl','wb'))


    # sympset-symp graph
    sympset_symp_graph = np.zeros([len(symp_set_set), len(symp_set)])
    sympset_symp_pair = []
    for sympsetid, sympset in setvoc['id2sympset'].items():
        for symp in sympset:
            sympset_symp_graph[sympsetid][symp] = 1
            sympset_symp_pair.append((sympsetid,symp,1)) 
    print(sympset_symp_graph.shape)
    dill.dump(sympset_symp_graph, open('sympset_symp_graph.pkl','wb'))
    dill.dump(sympset_symp_pair, open('sympset_symp_pair.pkl','wb'))

