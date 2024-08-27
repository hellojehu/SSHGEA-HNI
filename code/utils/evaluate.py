import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from .logreg import LogReg

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os.path
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

from sklearn.manifold import Isomap

# 节点嵌入聚类--聚类方法K-Means
def kmeans_cluster(embeds, labels, n_clusters):
    labels = labels.cpu().detach().numpy()
    embeds = embeds.cpu().detach().numpy()
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(embeds)
    p_labels = k_means.labels_
    nmi = normalized_mutual_info_score(labels, p_labels)
    ari = adjusted_rand_score(labels, p_labels)
    print('NMI Score: {:.5f}  |  ARI Score: {:.5f}'.format(nmi,ari))

def tsne_visualization(embeds, labels, dataset):
    if not os.path.exists("./new_visualization"):
        os.makedirs("./new_visualization")
    print("开始降维！！！")
    tsne = TSNE(n_components=2, random_state=0,perplexity=10,learning_rate=500,metric='euclidean')
    embeddings_tsne = tsne.fit_transform(embeds)
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', s=1)
    plt.colorbar()
    plt.savefig("./new_visualization/" + dataset + ".jpg")
    print("保存完成！！！")
    
def evaluate(embeds, idx_train, idx_val, idx_test, labels, num_class,isTest=True):
    print("----------------------------strat evaluating----------------------------------")
    hid_units = embeds.shape[1]
    nb_classes = num_class
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]
    train_lbls =labels[idx_train]
    val_lbls = labels[idx_val]
    test_lbls = labels[idx_test]

    run_num=10
    epoch_num=50
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    micro_f1s_val=[]
    for _ in range(run_num):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        val_accs = []; test_accs = []
        val_micro_f1s = []; test_micro_f1s = []
        val_macro_f1s = []; test_macro_f1s = []

        for iter_ in range(epoch_num):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward(retain_graph=True)
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter]) ###


        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
        micro_f1s_val.append(val_micro_f1s[max_iter])
    
    if isTest:
        print("\t[Classification-test] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s), np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s)))
