import time
from copy import deepcopy
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dhg import Graph
from dhg.data import Cora
from dhg.models import GCN
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
from sklearn.preprocessing import MinMaxScaler
from dhg import Graph, Hypergraph
from dhg.models import HGNNP

def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx].squeeze(), lbls[train_idx].float()
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res

def infer_test(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
        
    return res

def randbool(size, p=0.5):
    return torch.rand(*size) < p

def generate_boolean_vector(N, total_length):
    return np.array([True] * N + [False] * (total_length - N))

def dataset_create(data_excel,data_excel_Etest2_test):
    data_all = pd.read_excel(data_excel, sheet_name='Sheet1') #load all data
    
    data_os = data_all.iloc[1:2,1:]
    data_os_status = data_all.iloc[2:3,1:].astype(float)
    data_feat = data_all.iloc[3:1221, 1:]

    data_os_01 = data_os.astype(float)
    data_feat_01 = data_feat.astype(float)
    len_train = data_feat_01.shape[1]
    
    # Test dataset preparation
    data_all_Etest2_test = pd.read_excel(data_excel_Etest2_test, sheet_name='Sheet1') #load all data
    data_os_Etest2_test = data_all_Etest2_test.iloc[1:2,1:]
    data_os_status_Etest2_test = data_all_Etest2_test.iloc[2:3,1:].astype(float)
    data_feat_Etest2_test = data_all_Etest2_test.iloc[3:1221, 1:]
    

    data_os_Etest2_test = data_os_Etest2_test.astype(float)
    data_feat_Etest2_test = data_feat_Etest2_test.astype(float)
    len_Etest2_test = data_feat_Etest2_test.shape[1]
    
    data_feat_01 = np.hstack((data_feat_01, data_feat_Etest2_test))
    data_os_01 = np.hstack((data_os_01, data_os_Etest2_test))

    length = data_feat_01.shape[1]

    trn_idxes = generate_boolean_vector(len_train, length)
    tst_idxes = ~trn_idxes
    
    data_feat_numpy = data_feat_01
    data_feat_numpy_all =  torch.from_numpy(np.transpose(data_feat_numpy).astype('float32'))
    # Fill in empty values with the median of each column
    col_medians = np.nanmedian(data_feat_numpy_all, axis=0)  
    array = np.where(np.isnan(data_feat_numpy_all), col_medians, data_feat_numpy_all) 
    # Normalization by column
    scaler = MinMaxScaler()
    data_feat_numpy_all = torch.from_numpy(scaler.fit_transform(array))
    data_os_01 = np.where(np.isnan(data_os_01), np.nanmedian(data_os_01), data_os_01) 
    feature_dim = data_feat_numpy_all.shape[1]
    cutoff = "Median os of your dataset" #Create the true label of DHGN train and test
    data_os_lab = data_os_01.squeeze()
    data_os_label = [0 if a_ < cutoff else 1 for a_ in data_os_lab]
    data_os_label = torch.tensor(data_os_label)
    edge_list = "edge_list of the train and test dataset from PAE"
    dict_oc = {"features": data_feat_numpy_all, "edge_list": edge_list, "num_vertices": length, 
            "train_mask": trn_idxes, "test_mask":tst_idxes, "labels": data_os_label,"dim_features": feature_dim,
            "num_classes":1,'num_edges':len(edge_list)}
    return dict_oc
    
if __name__ == "__main__":
    set_seed(2025)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    
    data_ici = dataset_create("Your external train dataset excel", "Your external test dataset excel")# Etest2 as example
    data = data_ici
    all_feature, all_label = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]
    net = HGNNP(data["dim_features"], 16, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    all_feature, all_label = all_feature.to(device), all_label.to(device)
    G = G.to(device)
    HG = Hypergraph.from_graph(G)
    HG.add_hyperedges_from_graph_kHop(G, k=1)
    net = net.to(device)
    
    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(300):
        # train
        train(net, all_feature, HG, all_label, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, all_feature, HG, all_label, test_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state) ##Save the best GCN model
    res = infer(net, all_feature, HG, all_label, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    
    print("saving model...")
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    params_path = os.path.join(save_dir, "best_model_params.pth")
    torch.save(best_state, params_path)
    print(f"Model saved to {save_dir}")
