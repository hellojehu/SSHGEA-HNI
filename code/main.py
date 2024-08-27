import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import load_data, set_params
from utils.evaluate import evaluate, kmeans_cluster,tsne_visualization
from utils.cluster import kmeans
from module.att_lpa import *
from module.att_hgcn import ATT_HGCN
import warnings
import pickle as pkl
import os
import random
import time
warnings.filterwarnings('ignore')

import networkx as nx

dataset="imdb"
args = set_params(dataset)
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
train_percent=args.train_percent

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train():
    epochs =args.epochs 
    label, ft_dict, adj_dict = load_data(dataset,train_percent)
    G = nx.read_edgelist('./imdb.txt')
    
    target_type = args.target_type
    num_cluster = int(ft_dict[target_type].shape[0]*args.compress_ratio) # compress the range of initial pseudo-labels.
    num_class = np.unique(label[target_type][0]).shape[0]
    init_pseudo_label=0
    pseudo_pseudo_label = 0

    print('number of classes: ', num_cluster, '\n')
    layer_shape = []
    input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
    hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in args.hidden_dim]
    output_layer_shape = dict.fromkeys(ft_dict.keys(), num_cluster)

    layer_shape.append(input_layer_shape)
    layer_shape.extend(hidden_layer_shape)
    layer_shape.append(output_layer_shape)
    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    
    model = ATT_HGCN(
        net_schema=net_schema,
        layer_shape=layer_shape,
        label_keys=list(label.keys()),
        type_fusion=args.type_fusion,
        type_att_size=args.type_att_size,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if args.cuda and torch.cuda.is_available():
        model.cuda()
        for k in ft_dict:
            ft_dict[k] = ft_dict[k].cuda()
        for k in adj_dict:
            for kk in adj_dict[k]:
                adj_dict[k][kk] = adj_dict[k][kk].cuda()
        for k in label:
            for i in range(len(label[k])):
                label[k][i] = label[k][i].cuda()
    
    best = 1e9
    loss_list=[]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embd, attention_dict = model(ft_dict, adj_dict)
        target_embd = embd[target_type]
        if epoch==0:
            init_pseudo_label = init_lpa(adj_dict,ft_dict,target_type,num_cluster)
            pseudo_label_dict = init_pseudo_label
        elif epoch < args.warm_epochs:
            pseudo_label_dict=init_pseudo_label
        else:
            pseudo_label_dict = att_lpa(adj_dict,init_pseudo_label,attention_dict,target_type,num_cluster)
            init_pseudo_label=pseudo_label_dict
        label_predict=torch.argmax(pseudo_label_dict[target_type], dim=1)
        if epoch <= 100:
            m_label_predict = torch.argmax(pseudo_label_dict["m"], dim=1)
            a_label_predict = torch.argmax(pseudo_label_dict["a"], dim=1)
            u_label_predict = torch.argmax(pseudo_label_dict["u"], dim=1)
            d_label_predict = torch.argmax(pseudo_label_dict["d"], dim=1)
            m_nodes = list(n for n in G.nodes() if(n[0] == "m"))
            a_nodes = list(n for n in G.nodes() if(n[0] == "a"))
            u_nodes = list(n for n in G.nodes() if(n[0] == "u"))
            d_nodes = list(n for n in G.nodes() if(n[0] == "d"))
            for _ in range(1):
                for v in a_nodes:
                    v_list = list(G[v].keys())
                    dict_type = {}
                    flig = 0
                    for i in v_list:
                        index_v = int(i.split("m")[-1])
                        label_v = m_label_predict[index_v]
                        if label_v not in dict_type.keys():
                            dict_type[label_v] = 1
                        else:
                            dict_type[label_v] = dict_type[label_v] + 1
                            flig += 1
                        if flig == 0:
                            continue
                        else:
                            values = list(dict_type.values())
                            max_value = max(values)
                            max_key = next(key for key, value in dict_type.items() if value == max_value)
                            index = int(v.split("a")[-1])
                            a_label_predict[index] = max_key
                for v in u_nodes:
                    v_list = list(G[v].keys())
                    dict_type = {}
                    flig = 0
                    for i in v_list:
                        index_v = int(i.split("m")[-1])
                        label_v = m_label_predict[index_v]
                        if label_v not in dict_type.keys():
                            dict_type[label_v] = 1
                        else:
                            dict_type[label_v] = dict_type[label_v] + 1
                            flig += 1
                        if flig == 0:
                            continue
                        else:
                            values = list(dict_type.values())
                            max_value = max(values)
                            max_key = next(key for key, value in dict_type.items() if value == max_value)
                            index = int(v.split("u")[-1])
                            u_label_predict[index] = max_key
                for v in d_nodes:
                    v_list = list(G[v].keys())
                    dict_type = {}
                    flig = 0
                    for i in v_list:
                        index_v = int(i.split("m")[-1])
                        label_v = m_label_predict[index_v]
                        if label_v not in dict_type.keys():
                            dict_type[label_v] = 1
                        else:
                            dict_type[label_v] = dict_type[label_v] + 1
                            flig += 1
                        if flig == 0:
                            continue
                        else:
                            values = list(dict_type.values())
                            max_value = max(values)
                            max_key = next(key for key, value in dict_type.items() if value == max_value)
                            index = int(v.split("d")[-1])
                            d_label_predict[index] = max_key
            num_list = range(50000)
            for i,label_m in zip(num_list,m_label_predict):
                if i==0:
                    init_pseudo_label_m = torch.zeros(num_cluster)
                    init_pseudo_label_m[label_m] = 1
                    init_pseudo_label_m = init_pseudo_label_m.view(1, 193)
                else:
                    x = torch.zeros(num_cluster)
                    x[label_m] = 1
                    x = x.view(1, 193)
                    init_pseudo_label_m = torch.cat((init_pseudo_label_m, x))
            for i,label_a in zip(num_list,a_label_predict):
                if i==0:
                    init_pseudo_label_a = torch.zeros(num_cluster)
                    init_pseudo_label_a[label_a] = 1
                    init_pseudo_label_a = init_pseudo_label_a.view(1, 193) # 可以将193换成num_cluster
                else:
                    x = torch.zeros(num_cluster)
                    x[label_a] = 1
                    x = x.view(1, 193)
                    init_pseudo_label_a = torch.cat((init_pseudo_label_a, x))
            for i,label_u in zip(num_list,u_label_predict):
                if i==0:
                    init_pseudo_label_u = torch.zeros(num_cluster)
                    init_pseudo_label_u[label_u] = 1
                    init_pseudo_label_u = init_pseudo_label_u.view(1, 193)
                else:
                    x = torch.zeros(num_cluster)
                    x[label_u] = 1
                    x = x.view(1, 193)
                    init_pseudo_label_u = torch.cat((init_pseudo_label_u, x))
            for i,label_d in zip(num_list,d_label_predict):
                if i==0:
                    init_pseudo_label_d = torch.zeros(num_cluster)
                    init_pseudo_label_d[label_d] = 1
                    init_pseudo_label_d = init_pseudo_label_d.view(1, 193)
                else:
                    x = torch.zeros(num_cluster)
                    x[label_d] = 1
                    x = x.view(1, 193)
                    init_pseudo_label_d = torch.cat((init_pseudo_label_d, x))
            init_pseudo_label_new = {}
            init_pseudo_label_new["m"] = init_pseudo_label_m
            init_pseudo_label_new["d"] = init_pseudo_label_d
            init_pseudo_label_new["u"] = init_pseudo_label_u
            init_pseudo_label_new["a"] = init_pseudo_label_a
            init_pseudo_label.clear()
            init_pseudo_label = init_pseudo_label_new
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for key in init_pseudo_label:
                init_pseudo_label[key] = init_pseudo_label[key].to(device)
        logits = F.log_softmax(logits[target_type], dim=1)
        loss_train = F.nll_loss(logits, label_predict.long().detach())
        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train)
        if loss_train < best:
            best = loss_train

        print(
            'epoch: {:3d}'.format(epoch),
            'train loss: {:.4f}'.format(loss_train.item()),
        )

    #evaluate
    logits, embd, _ = model(ft_dict, adj_dict)
    target_embd = embd[target_type]
    label_target = label[target_type]
    true_label = label_target[0]
    idx_train = label_target[1]
    idx_val = label_target[2]
    idx_test = label_target[3]

    kmeans_cluster(target_embd,true_label,args.n_clusters)
    tsne_visualization(target_embd,true_label,dataset)
    evaluate(target_embd, idx_train, idx_val, idx_test, true_label, num_class, isTest=True)

if __name__ == '__main__':
    train()
