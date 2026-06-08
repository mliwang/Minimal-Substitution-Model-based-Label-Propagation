#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import *
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TransformerConv, SGConv, LEConv, GENConv, \
    FiLMConv, TAGConv
from models import *
from torch_geometric.utils import to_undirected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat = [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
import warnings

warnings.filterwarnings("ignore")



use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(use_device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # **Please insert Kaggle username and kaggle key**


def make_data(datadir="../dataset/", tag="mkdat",use_labelbalance='msm'):
    # if os.path.exists('dat_' + f'{tag}'):
    #     with open('dat_' + f'{tag}', "rb") as f:
    #         ALL_DATA = pickle.load(f)
    #     return ALL_DATA

    # Load Dataframe
    df_edge = pd.read_csv(datadir + 'elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    df_class = pd.read_csv(datadir + 'elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    df_features = pd.read_csv(datadir + 'elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)

    # print(df_features)

    # Setting Column name
    df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                          range(72)]

    print('Number of edges: {}'.format(len(df_edge)))

    # 将三个数据集中所有的节点合并到一个集合中
    all_nodes = list(
        set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
    # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。将集合all_nodes置位DataFrame，列名为'id'，行名保留，同时置位0,1,2...
    nodes_df = pd.DataFrame(all_nodes, columns=['id']).reset_index()
    NUM_NODES = len(nodes_df)
    print('Number of nodes: {}'.format(NUM_NODES))
    NUM_FEAT = 166

    # ## Fix id index
    # 234355 rows x 2 columns
    df_edge = df_edge.join(nodes_df.rename(columns={'id': 'txId1'}).set_index('txId1'), on='txId1', how='inner').join(
        nodes_df.rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='inner', rsuffix='2').drop(
        columns=['txId1', 'txId2']).rename(columns={'index': 'txId1', 'index2': 'txId2'})
    df_edge.head()

    # 203769 rows x 2 columns
    df_class = df_class.join(nodes_df.rename(columns={'id': 'txId'}).set_index('txId'), on='txId', how='inner').drop(
        columns=['txId']).rename(columns={'index': 'txId'})[['txId', 'class']]
    df_class.head()

    # 203769 rows x 167 columns
    df_features = df_features.join(nodes_df.set_index('id'), on='id', how='inner').drop(columns=['id']).rename(
        columns={'index': 'id'})
    df_features = df_features[['id'] + list(df_features.drop(columns=['id']).columns)]
    df_features.head()

    df_edge_time = df_edge.join(df_features[['id', 'time step']].rename(columns={'id': 'txId1'}).set_index('txId1'),
                                on='txId1', how='left', rsuffix='1').join(
        df_features[['id', 'time step']].rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='left',
        rsuffix='2')
    df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
    # 234355 rows x 3 columns
    df_edge_time_fin = df_edge_time[['txId1', 'txId2', 'time step']].rename(
        columns={'txId1': 'source', 'txId2': 'target', 'time step': 'time'})

    # ## Create csv from Dataframe

    # # 删除索引为'time step'这一列
    # df_features.drop(columns=['time step']).to_csv('elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv',
    #                                                index=False,
    #                                                header=None)
    # df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(by='nid').to_csv(
    #     'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv', index=False, header=None)
    # df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'})[['nid', 'time']].sort_values(
    #     by='nid').to_csv('elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv', index=False, header=None)
    # df_edge_time_fin[['source', 'target', 'time']].to_csv(
    #     'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
    #     index=False, header=None)

    # ## Graph Preprocessing

    node_label = df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(
        by='nid').merge(
        df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'}), on='nid', how='left')
    node_label['label'] = node_label['label'].apply(lambda x: '3' if x == 'unknown' else x).astype(int) - 1
    node_label.head()
    # print("***", node_label['label'].unique())
    # print("node_label:", node_label)

    merged_nodes_df = node_label.merge(
        df_features.rename(columns={'id': 'nid', 'time step': 'time'}).drop(columns=['time']), on='nid', how='left')
    merged_nodes_df.head()
    merged_nodes_df.to_csv("original.csv",index=False)
    # print("merged_nodes_df:", merged_nodes_df)

    # print(df_features)
    # group_feature = [0 for index in range(168)]
    # # print(group_feature)
    # #统计原始特征和统计特征数量
    # for i in range(93):
    #     group_feature[i] = df_features.groupby('trans_feat_'+str(i)).count()
    #     group_feature[i]['id'].plot()
    #     plt.title('Number of transactions by trans_feat_'+str(i))
    #     plt.savefig(fname='trans_feat_'+str(i))
    #     plt.show()
    # for i in range(72):
    #     group_feature[i+94] = df_features.groupby('agg_feat_' + str(i)).count()
    #     group_feature[i+94]['id'].plot()
    #     plt.title('Number of transactions by agg_feat_' + str(i))
    #     plt.savefig(fname='agg_feat_' + str(i))
    #     plt.show()
    # 拿到所有交易节点出现的时间
    saa = []
    for name, df in merged_nodes_df.groupby(merged_nodes_df["nid"]):
        df.sort_values(by="time", inplace=True, ascending=True)
        saa.append(df[["nid", "time"]])

    node_proTime = pd.concat(saa, axis=0, ignore_index=True)  # 节点产生时间

    train_dataset = []
    test_dataset = []
    # test_dataset2 = []
    for i in range(49):
        nodes_df_tmp = merged_nodes_df[merged_nodes_df['time'] == i + 1].reset_index()
        nodes_df_tmp['index'] = nodes_df_tmp.index
        df_edge_tmp = df_edge_time_fin.join(
            nodes_df_tmp.rename(columns={'nid': 'source'})[['source', 'index']].set_index('source'), on='source',
            how='inner').join(nodes_df_tmp.rename(columns={'nid': 'target'})[['target', 'index']].set_index('target'),
                              on='target', how='inner', rsuffix='2').drop(columns=['source', 'target']).rename(
            columns={'index': 'source', 'index2': 'target'})
        if use_labelbalance is None:
            pass
        if use_labelbalance=="msm":
            nodes_df_tmp = transferMSML(nodes_df_tmp, node_proTime, df_edge_tmp[['source', 'target']])
        elif use_labelbalance=="random":
            nodes_df_tmp = RandomtransferL(nodes_df_tmp)
        elif use_labelbalance=="sim":
            nodes_df_tmp = simtransferL1(nodes_df_tmp, node_proTime,df_edge_tmp[['source', 'target']])
        elif use_labelbalance=="lp":
            nodes_df_tmp = transferlabelPropagation(nodes_df_tmp)



        # nodes_df_tmp = transferL(nodes_df_tmp, 800)

        # nodes_df_tmp =RandomtransferL(nodes_df_tmp)
        # nodes_df_tmp =simtransferL1(nodes_df_tmp, node_proTime,df_edge_tmp[['source', 'target']])

        x = torch.tensor(np.array(nodes_df_tmp.sort_values(by='index').drop(columns=['index', 'nid', 'label'])),
                         dtype=torch.float)
        # if(i == 1):
        #     x = x.unsqueeze(0)
        #     print("---", x)
        # x = x.unsqueeze(0)
        # print(x.shape)
        edge_index = torch.tensor(np.array(df_edge_tmp[['source', 'target']]).T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        mask = nodes_df_tmp['label'] != 2
        y = torch.tensor(np.array(nodes_df_tmp['label']))
        # print(y.shape)
        # 划分测试集
        if i + 1 < 35:
            data = Data(x=x, edge_index=edge_index, train_mask=mask, y=y)
            train_dataset.append(data)
        else:
            data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
            test_dataset.append(data)

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True)

    # test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)
    ALL_DATA = [NUM_NODES, NUM_FEAT, train_loader, test_loader]
    # pickle.dump(ALL_DATA, open('dat_' + f'{tag}', 'wb'))
    # with open('dat_' + f'{tag}', "rb") as f:
    #     ALL_DATA = pickle.load(f)
    return ALL_DATA


# #### Hyperparameter


def tain_model(train_loader, test_loader, model, use_criterion, epoches=300):
    lr = 0.001
    epoches = epoches

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # criterion = use_criterion
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))
    # criterion = F.nll_loss
    train_losses = []
    val_losses = []
    accuracies = []
    if1 = []
    precisions = []
    recalls = []
    AUCs = []
    iterations = []

    # logf = f"log_{tag}"
    all_logstr = ""
    for epoch in range(epoches):

        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].long())
            _, pred = out[data.train_mask].max(dim=1)
            loss.backward()
            train_loss += loss.item() * data.num_graphs
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        # 评估验证集
        if (epoch + 1) % 2 == 0:
            model.eval()
            ys, preds = [], []
            val_loss = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out[data.test_mask], data.y[data.test_mask].long())
                val_loss += loss.item() * data.num_graphs
                _, pred = out[data.test_mask].max(dim=1)
                ys.append(data.y[data.test_mask].cpu())
                preds.append(pred.cpu())

            y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
            val_loss /= len(test_loader.dataset)
            f1 = f1_score(y, pred, average=None)
            mf1 = f1_score(y, pred, average='macro')
            precision = precision_score(y, pred, average=None)
            recall = recall_score(y, pred, average=None)
            # AUC = roc_auc_score(y, pred)

            iterations.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if1.append(f1[0])
            accuracies.append(mf1)
            precisions.append(precision[0])
            recalls.append(recall[0])
            # AUCs.append(AUC)#计算AUC
            # fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)#计算ROC
            log_str = 'Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f},非法 Precision: {:.4f}, Recall: {:.4f}, Illicit f1: {:.4f}, mF1: {:.4f}'.format(
                epoch + 1, train_loss, val_loss, precision[0], recall[0], f1[0], mf1) + "\n" + \
                      f'合法 class precision: {precision[1]}, recall: {recall[1]}, f1: {f1[1]}\n'
            # log_str = 'Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Illicit f1: {:.4f}'.format(
            #     epoch + 1, train_loss, val_loss, precision[0], recall[0], f1[0]) + "\n"
            print(log_str
                  )
            # fi.write((f"{val_loss}\n"))
            # if epoch+1==10:
            #     fi.write(f"{precision[0], recall[0], f1[0]}\n")
            all_logstr += log_str

    # with open(logf, "w+") as f:
    #     f.write(all_logstr)
    #
    # a, b, c, d = train_losses, val_losses, if1, accuracies

    # import pickle
    #
    # g = [a, b, c, d]
    # pickle.dump(g, open('res_' + f'{tag}', 'wb'))
    # with open('res_' + f'{tag}', "rb") as f:
    #     g = pickle.load(f)
    # a, b, c, d = g

    # ep = [i for i in range(patience, epoches + 1, patience)]


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, conv1=GCNConv, conv2=GCNConv, use_skip=False):
        super(GCN, self).__init__()
        self.conv1 = conv1(num_node_features, hidden_channels[0])
        self.conv2 = conv2(hidden_channels[0], 2)
        self.use_skip = use_skip

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x


# run using GCN model
if __name__ == "__main__":
    # GCNN = TGCN
    NUM_NODES, NUM_FEAT, train_loader, test_loader = make_data(use_labelbalance="msm")
    epoches = 200

    print(f"make_data ok!!!, epoches = {epoches}")
    # fi = open('./result_GCN.txt', 'w')

    # tag,
    # for i in [(GCNConv,GCNConv,False),(GATConv,GATConv,False)]:
    #     conv1, conv2, useskip = i
    #
    #     model = GCN2layer(NUM_FEAT, [100], conv1, conv1, use_skip=useskip)
    #     model.to(device)
    #     lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]).to(device))
    #     tain_model(train_loader, test_loader, model, lossf, epoches=epoches)
    conv1, conv2, useskip = FiLMConv, FiLMConv, False

    model = GCN(NUM_FEAT, [256], conv1, conv1, use_skip=useskip)
    model.to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]).to(device))
    tain_model(train_loader, test_loader, model, lossf, epoches=epoches)
    # torch.save(model.state_dict(), f'model_{tag}.pkl')
    # break
    # 加载
    # model = torch.load(f'\model_{tag}.pkl')
    # model.load_state_dict(torch.load('\parameter.pkl'))




