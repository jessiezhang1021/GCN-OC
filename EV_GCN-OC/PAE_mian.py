import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np
import xlrd
import pandas as pd
import datetime
 

from opt import * 
from EV_GCN_PAE import EV_GCN
from utils.metrics_PAE import accuracy, auc, prf
from dataloader_PAE import dataloader 
from PAE import PAE


def dataset_create(data_excel):
    
    data_all = pd.read_excel(data_excel, sheet_name='Sheet1')#load data
    length = data_all.shape[1]
    data_feat = data_all.iloc["Your feature columns in the excel", 1:]

    Etest1_Index = data_feat.columns[data_feat.columns.str.contains('Your external test 1 dataset', case=False)]
    Etest2_Index = data_feat.columns[data_feat.columns.str.contains('Your external test 1 dataset', case=False)]
    data_feat_test_Etest1_Index = data_feat[Etest1_Index]
    data_feat_test_Etest2_Index =  data_feat[Etest2_Index]
    
    #feature filter
    all_columns = data_feat.columns
    b_columns = data_feat_test_Etest1_Index.columns
    c_columns = data_feat_test_Etest2_Index.columns
    remaining_columns = all_columns.difference(b_columns.union(c_columns))
    selected_indices = list(range(1, 3))
    selected_array = remaining_columns[selected_indices]
    remaining_array = np.delete(remaining_columns, selected_indices)
    
    data_feat_train = data_feat[remaining_array]
    data_feat_train_numpy = data_feat_train.to_numpy()
    data_feat_train_numpy =  np.transpose(data_feat_train_numpy).astype('float32')
    col_medians = np.nanmedian(data_feat_train_numpy, axis=0) 
    array = np.where(np.isnan(data_feat_train_numpy), col_medians, data_feat_train_numpy)  # Fill in empty values with median
    nonimg_train = array
    
    data_feat_inter_test = data_feat[selected_array]
    data_feat_inter_test_numpy = data_feat_inter_test.to_numpy()
    data_feat_inter_test_numpy =  np.transpose(data_feat_inter_test_numpy).astype('float32')
    col_medians = np.nanmedian(data_feat_inter_test_numpy, axis=0)  
    array = np.where(np.isnan(data_feat_inter_test_numpy), col_medians, data_feat_inter_test_numpy)   # Fill in empty values with median
    nonimg_inter_test = array
    
    data_feat_train_all = data_feat[remaining_columns]
    data_feat_train_all_numpy = data_feat_train_all.to_numpy()
    data_feat_train_all_numpy =  np.transpose(data_feat_train_all_numpy).astype('float32')
    col_medians = np.nanmedian(data_feat_train_all_numpy, axis=0)  
    array = np.where(np.isnan(data_feat_train_all_numpy), col_medians, data_feat_train_all_numpy)   # Fill in empty values with median
    nonimg_train_all = array
    
    data_feat_test_Etest1_numpy = data_feat_test_Etest1_Index.to_numpy()
    data_feat_test_Etest1_numpy =  np.transpose(data_feat_test_Etest1_numpy).astype('float32')
    col_medians = np.nanmedian(data_feat_test_Etest1_numpy, axis=0) 
    array = np.where(np.isnan(data_feat_test_Etest1_numpy), col_medians, data_feat_test_Etest1_numpy)   # Fill in empty values with median
    nonimg_Etest1 = array
    
    data_feat_test_Etest2_numpy = data_feat_test_Etest2_Index.to_numpy()
    data_feat_test_Etest2_numpy =  np.transpose(data_feat_test_Etest2_numpy).astype('float32')
    col_medians = np.nanmedian(data_feat_test_Etest2_numpy, axis=0)  
    array = np.where(np.isnan(data_feat_test_Etest2_numpy), col_medians, data_feat_test_Etest2_numpy)   # Fill in empty values with median
    nonimg_Etest2 = array
    
    data_os = data_all.iloc[["Your survivial column"], 1:]
    data_os_lab_Etest1 = data_os[Etest1_Index]
    data_os_lab_Etest2 = data_os[Etest2_Index]
    data_os_lab_train_all = data_os[remaining_columns]
    data_os_lab_train = data_os[remaining_array]
    data_os_lab_inter_test = data_os[selected_array]
    data_os_lab_Etest1 = data_os_lab_Etest1.to_numpy().squeeze().astype('float32')
    data_os_lab_Etest2 = data_os_lab_Etest2.to_numpy().squeeze().astype('float32')
    data_os_lab_train = data_os_lab_train.to_numpy().squeeze().astype('float32')
    data_os_lab_inter_test = data_os_lab_inter_test.to_numpy().squeeze().astype('float32')
    data_os_lab_train_all = data_os_lab_train_all.to_numpy().squeeze().astype('float32')
    
    data_os_lab_Etest1 = np.where(np.isnan(data_os_lab_Etest1), np.nanmedian(data_os_lab_Etest1), data_os_lab_Etest1)   # Fill in empty values with median
    data_os_lab_Etest2 = np.where(np.isnan(data_os_lab_Etest2), np.nanmedian(data_os_lab_Etest2), data_os_lab_Etest2)
    data_os_lab_train = np.where(np.isnan(data_os_lab_train), np.nanmedian(data_os_lab_train), data_os_lab_train)
    data_os_lab_inter_test = np.where(np.isnan(data_os_lab_inter_test), np.nanmedian(data_os_lab_inter_test), data_os_lab_inter_test)
    data_os_lab_train_all = np.where(np.isnan(data_os_lab_train_all), np.nanmedian(data_os_lab_train_all), data_os_lab_train_all)

    
    return data_os_lab_Etest1, data_os_lab_Etest2, data_os_lab_train, data_os_lab_inter_test, data_os_lab_train_all,\
        nonimg_Etest1, nonimg_Etest2, nonimg_train, nonimg_inter_test, nonimg_train_all

if __name__ == '__main__':
    
    opt = OptInit().initialize()
    dl = dataloader()
    raw_features, y, nonimg = dl.load_data()
    n_folds = N #"N-fold"
    cv_splits = dl.data_split(n_folds)

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)

    PFS_Etest1, PFS_Etest2, PFS_train, PFS_inter_test, PFS_train_all, nonimg_Etest1, nonimg_Etest2, nonimg_train, nonimg_inter_test, nonimg_train_all \
    = dataset_create("Your excel in you local computer")


    print('  Constructing graph data...')
    edge_label, edge_index, edgenet_input = dl.get_PAE_inputs(nonimg_train_all, PFS_train_all)
    edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
    edge_label_inter_test, edge_index_inter_test, edgenet_input_inter_test = dl.get_PAE_inputs(nonimg_inter_test, PFS_inter_test) 
    edgenet_input_inter_test = (edgenet_input_inter_test - edgenet_input_inter_test.mean(axis=0)) / edgenet_input_inter_test.std(axis=0)
    
    
    edge_label_test_Etest2, edge_index_test_Etest2, edgenet_input_test_Etest2 = dl.get_PAE_inputs(nonimg_Etest2, PFS_Etest2) 
    edgenet_input_test_Etest2 = (edgenet_input_test_Etest2 - edgenet_input_test_Etest2.mean(axis=0)) / edgenet_input_test_Etest2.std(axis=0)
    edge_label_test_Etest1, edge_index_test_Etest1, edgenet_input_test_Etest1 = dl.get_PAE_inputs(nonimg_Etest1, PFS_Etest1) 
    edgenet_input_test_Etest1 = (edgenet_input_test_Etest1 - edgenet_input_test_Etest1.mean(axis=0)) / edgenet_input_test_Etest1.std(axis=0)
    model = EV_GCN(opt.dropout, edge_dropout=opt.edropout, edgenet_input_dim=2*nonimg_train.shape[1]).to(opt.device)
    model = model.to(opt.device)

    loss_fn =torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
    edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
    
    edge_index_inter_test_1 = torch.tensor(edge_index_inter_test, dtype=torch.long).to(opt.device)
    edgenet_input_inter_test_1 = torch.tensor(edgenet_input_inter_test, dtype=torch.float32).to(opt.device)
    
    labels = torch.tensor(edge_label, dtype=torch.float32).to(opt.device)
    labels_inter_test = torch.tensor(edge_label_inter_test, dtype=torch.float32).to(opt.device)
    now = datetime.datetime.now()
    fold_model_path = opt.ckpt_path + "/{}.pth".format(now)
    
    print("  Number of training samples %d" % len(PFS_train))
    print("  Start training...\r\n")
    acc = 0

    
    if opt.train==1:
        for epoch in range(opt.num_iter):
            model.train()  
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                edge_weights = model(edge_index, edgenet_input)
                #loss = loss_fn(node_logits[train_ind], labels[train_ind])
                loss = loss_fn(edge_weights, labels)
                loss.backward()
                optimizer.step()
                print("Epoch: {},\tce loss: {:.5f}".format(epoch, loss.item()))
                edge_weights_pred_labels = torch.where(edge_weights > 0.5, 1, 0) ##Note: This cut-off could be determined based on your data
            correct_train, acc_train = accuracy(edge_weights_pred_labels.detach().cpu().numpy(), labels.detach().cpu().numpy())
            
            model.eval()
            with torch.set_grad_enabled(False):
                edge_weights_test = model(edge_index_inter_test_1, edgenet_input_inter_test_1)
            edge_weights_test_pred_labels = torch.where(edge_weights_test > 0.5, 1, 0)
            edge_weights_test_pred_labels = edge_weights_test_pred_labels.detach().cpu().numpy()
            correct_test, acc_test = accuracy(edge_weights_test_pred_labels, labels_inter_test.detach().cpu().numpy())

            if acc_test > acc and epoch >90:
                print(acc_test)
                acc = acc_test
                correct = correct_test 
                if opt.ckpt_path !='':
                    if not os.path.exists(opt.ckpt_path): 
                        os.makedirs(opt.ckpt_path)
                    torch.save(model.state_dict(), fold_model_path)
        opt.train = 0
        
    if opt.train==0:
        print('  Start testing...')
        model.load_state_dict(torch.load(fold_model_path)) 
        model.eval()
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
    
        labels = torch.tensor(edge_label, dtype=torch.float32).to(opt.device)
        
        edge_index_inter_test = torch.tensor(edge_index_inter_test, dtype=torch.long).to(opt.device)
        edgenet_input_inter_test = torch.tensor(edgenet_input_inter_test, dtype=torch.float32).to(opt.device)
        labels_inter_test = torch.tensor(edge_label_inter_test, dtype=torch.float32).to(opt.device)
        
        edge_index_test_Etest2 = torch.tensor(edge_index_test_Etest2, dtype=torch.long).to(opt.device)
        edgenet_input_test_Etest2 = torch.tensor(edgenet_input_test_Etest2, dtype=torch.float32).to(opt.device)
        labels_test_Etest2 = torch.tensor(edge_label_test_Etest2, dtype=torch.float32).to(opt.device)
        
        edge_index_test_Etest1 = torch.tensor(edge_index_test_Etest1, dtype=torch.long).to(opt.device)
        edgenet_input_test_Etest1 = torch.tensor(edgenet_input_test_Etest1, dtype=torch.float32).to(opt.device)
        labels_test_Etest1 = torch.tensor(edge_label_test_Etest1, dtype=torch.float32).to(opt.device)
        
        edge_weights = model(edge_index, edgenet_input)
        edge_weights_train_pred_labels = torch.where(edge_weights > "Your own cut-off based on your dataset or use 0.5 directly", 1, 0)
        edge_weights_train_pred_labels = edge_weights_train_pred_labels.detach().cpu().numpy()
        correct_train, acc_train = accuracy(edge_weights_train_pred_labels, labels.detach().cpu().numpy())
        aucs_train = auc(edge_weights_train_pred_labels,labels.detach().cpu().numpy()) 
        prfs_train  = prf(edge_weights_train_pred_labels,labels.detach().cpu().numpy())  
        

        edge_weights = model(edge_index, edgenet_input)
        edge_weights_train_pred_labels = torch.where(edge_weights > "Your own cut-off based on your dataset or use 0.5 directly", 1, 0)
        edge_weights_train_pred_labels = edge_weights_train_pred_labels.detach().cpu().numpy()
        correct_train, acc_train = accuracy(edge_weights_train_pred_labels, labels.detach().cpu().numpy())
        aucs_train = auc(edge_weights_train_pred_labels,labels.detach().cpu().numpy()) 
        prfs_train  = prf(edge_weights_train_pred_labels,labels.detach().cpu().numpy())
        se, sp, f1 = prfs_train

        
        edge_weights_test_Etest2 = model(edge_index_test_Etest2, edgenet_input_test_Etest2)
        edge_weights_test_Etest2_pred_labels = torch.where(edge_weights_test_Etest2 > "Your own cut-off based on your dataset or use 0.5 directly", 1, 0)
        edge_weights_test_Etest2_pred_labels = edge_weights_test_Etest2_pred_labels.detach().cpu().numpy()
        correct_test_Etest2, acc_test_Etest2 = accuracy(edge_weights_test_Etest2_pred_labels, labels_test_Etest2.detach().cpu().numpy())
        aucs_test_Etest2 = auc(edge_weights_test_Etest2_pred_labels,labels_test_Etest2.detach().cpu().numpy()) 
        prfs_test_Etest2  = prf(edge_weights_test_Etest2_pred_labels,labels_test_Etest2.detach().cpu().numpy())  
        se, sp, f1 = prfs_test_Etest2
        print("=> Average test Etest2 with acc {:.4f}, sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(acc_test_Etest2, se, sp, f1))




        edge_weights_test_Etest1 = model(edge_index_test_Etest1, edgenet_input_test_Etest1)
        edge_weights_test_Etest1_pred_labels = torch.where(edge_weights_test_Etest1 > "Your own cut-off based on your dataset or use 0.5 directly", 1, 0)
        edge_weights_test_Etest1_pred_labels = edge_weights_test_Etest1_pred_labels.detach().cpu().numpy()
        correct_test_Etest1, acc_test_Etest1 = accuracy(edge_weights_test_Etest1_pred_labels, labels_test_Etest1.detach().cpu().numpy())
        aucs_test_Etest1 = auc(edge_weights_test_Etest1_pred_labels,labels_test_Etest1.detach().cpu().numpy()) 
        prfs_Etest1  = prf(edge_weights_test_Etest1_pred_labels,labels_test_Etest1.detach().cpu().numpy())
        se, sp, f1 = prfs_Etest1
        print("=> Average test Etest1 with acc {:.4f}, sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(acc_test_Etest1, se, sp, f1))
