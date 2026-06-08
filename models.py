import torch
import warnings
from sklearn.cluster import KMeans
from LabelPropo import labelPropagation
import numpy as np
from numpy import *
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat=[f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
def tkmeans(alldata):
    '''
    kmeans聚类

    Returns
    -------
    None.

    '''
    clf = KMeans(n_clusters=2)
    ydata = clf.fit_predict(alldata)
    alldata['label']= clf.labels_
    return  alldata


def transferMSML(nodes_df_tmp, node_pro_time, Adj):
    '''

有平衡过多的风险
    Parameters
    ----------
    nodes_df_tmp : TYPE 需要做标签传播的dataframe
        DESCRIPTION.
    node_pro_time : TYPE  所有节点的产生时间
        DESCRIPTION.
    Adj : TYPE       对应的节点间的关系图
        DESCRIPTION.
均衡数 min(#正常交易节点数-#异常交易数量, 2η-k)
    Returns
    -------
    None.

    '''
    print("Orinal:", nodes_df_tmp['label'].value_counts())
    res_ids_unkown = nodes_df_tmp[nodes_df_tmp['label'] == 2]  # .drop(columns=['index', 'nid', 'label', 'time'])
    res_ids_other = nodes_df_tmp[
        (nodes_df_tmp['label'] == 0) | (nodes_df_tmp['label'] == 1)]  # .drop(columns=['index', 'nid', 'label', 'time'])
    res_ids_unkown_np = res_ids_unkown[feat].values
    res_ids_other_np = res_ids_other[feat].values
    res_ids_other_label = np.matrix(nodes_df_tmp[(nodes_df_tmp['label'] == 0) | (nodes_df_tmp['label'] == 1)]['label'])
    num_balence = res_ids_other[res_ids_other['label'] == 1]['nid'].nunique() - \
                  res_ids_other[res_ids_other['label'] == 0]['nid'].nunique()
    print("number of balance:", num_balence)
    from sklearn.metrics.pairwise import cosine_similarity

    # 1.求余弦相似度
    sim = cosine_similarity(res_ids_unkown_np, res_ids_other_np)  # K,M
    # print(sim)
    # 2.求入度矩阵  从known 到unknown
    indict = dict(Adj.groupby('target').size())

    res_ids_unkown['indegree'] = res_ids_unkown['nid'].map(indict)
    res_ids_unkown['indegree'].fillna(0, inplace=True)
    inDegree = res_ids_unkown['indegree'].values  # k,1
    inDegree = np.reshape(inDegree, (len(inDegree), 1))

    # 求η
    η = nodes_df_tmp['nid'].nunique() / (mean(inDegree)) ** 2
    num_balence = min(2 * η, num_balence)

    # 3.求对应的节点的时间特性

    time_key = res_ids_unkown[['nid'] + feat].join(node_pro_time.set_index('nid'), on='nid', how='inner')
    T = time_key['time'].values  # k,1
    T = np.reshape(T, (len(T), 1))
    # 4.MMS
    res_ids = sim * inDegree / T

    # 行归一化处理

    res_ids_max, res_ids_min, res_ids_mean = res_ids.max(axis=0), res_ids.min(axis=0), res_ids.mean(axis=0)
    res_ids = (res_ids - res_ids_mean) / (res_ids_max - res_ids_min)  # 此处需要处理成0-1

    res_ids_label = res_ids * res_ids_other_label.T  # K,1的矩阵
    # print("res_ids_label****",res_ids_label)
    resmean = res_ids_label.mean()

    # print("res_ids_label", res_ids_label)
    def change1(x):
        if x > 0.5:
            return 1
        else:
            return 0

    res_ids_unkown['label'] = res_ids_label  # 非0，1
    # 二值化
    res_ids_unkown['label'] = res_ids_unkown['label'].apply(change1)
    # global count
    # count = 0
    # 5.样本均衡
    canbalence = len(res_ids_unkown[res_ids_unkown['label'] == 0])
    if canbalence > num_balence:
        res_ids_unkown_1 = res_ids_unkown[res_ids_unkown['label'] == 0].sample(num_balence)  # 保留部分预测为0（异常）的
        # nodes_df_tmp.loc[nodes_df_tmp['nid'].isin(set(res_ids_unkown[res_ids_unkown['label']==0].nid.values)-set(res_ids_unkown_1.nid.values)),'label']=0
        res_ids_unkown = res_ids_unkown_1
    res_ids_unkown = res_ids_unkown[res_ids_unkown['label'] == 0]
    nodes_df_tmp.loc[nodes_df_tmp['nid'].isin(res_ids_unkown.nid), 'label'] = 0

    # nodes_df_tmp1 = nodes_df_tmp.apply(lambda x: change2(x), axis=1)
    print("After balance:", nodes_df_tmp['label'].value_counts())

    return nodes_df_tmp


def RandomtransferL(nodes_df_tmp):
    res_ids_unkown = nodes_df_tmp[nodes_df_tmp['label'] == 2]  # .drop(columns=['index', 'nid', 'label', 'time'])
    res_ids_other = nodes_df_tmp[
        (nodes_df_tmp['label'] == 0) | (nodes_df_tmp['label'] == 1)]  # .drop(columns=['index', 'nid', 'label', 'time'])
    num_balence = res_ids_other[res_ids_other['label'] == 1]['nid'].nunique() - \
                  res_ids_other[res_ids_other['label'] == 0]['nid'].nunique()
    res_ids_unkown = res_ids_unkown.sample(num_balence)
    nodes_df_tmp.loc[nodes_df_tmp['nid'].isin(res_ids_unkown.nid), 'label'] = 1
    return nodes_df_tmp


def transferlabelPropagation(nodes_df_tmp):
    '''


    Parameters
    ----------
    nodes_df_tmp : TYPE 需要做标签传播的dataframe
        DESCRIPTION.
    node_pro_time : TYPE  所有节点的产生时间
        DESCRIPTION.
    Adj : TYPE       对应的节点间的关系图
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    res_ids_unkown = nodes_df_tmp[nodes_df_tmp['label'] == 2]  # .drop(columns=['index', 'nid', 'label', 'time'])
    res_ids_other = nodes_df_tmp[(nodes_df_tmp['label'] == 0) | (nodes_df_tmp['label'] == 1)]
    num_balence = res_ids_other[res_ids_other['label'] == 0]['nid'].nunique() - \
                  res_ids_other[res_ids_other['label'] == 1]['nid'].nunique()
    unlabeled_points = res_ids_unkown['nid'].values
    # nodes_df_tmp.loc[nodes_df_tmp.nid.isin(unlabeled_points),'label']=-1
    labels = np.copy(nodes_df_tmp.label.values)
    print('Unlabeled Number:', list(labels).count(2))

    Y_pred = labelPropagation(res_ids_other[feat].values, res_ids_unkown[feat].values, res_ids_other['label'].values,
                              kernel_type='knn', knn_num_neighbors=10, max_iter=500)
    res_ids_unkown['label'] = Y_pred  # 非0，1
    # 5.样本均衡

    canbalence = len(res_ids_unkown[res_ids_unkown['label'] == 1])
    if canbalence > num_balence:
        res_ids_unkown_1 = res_ids_unkown[res_ids_unkown['label'] == 1].sample(num_balence)  # 保留部分预测为1（异常）的

        nodes_df_tmp.loc[nodes_df_tmp['nid'].isin(set(res_ids_unkown[res_ids_unkown['label'] == 1].nid.values) - set(
            res_ids_unkown_1.nid.values)), 'label'] = 0
        res_ids_unkown = res_ids_unkown_1
    res_ids_unkown = res_ids_unkown[res_ids_unkown['label'] == 1]
    nodes_df_tmp.loc[nodes_df_tmp['nid'].isin(res_ids_unkown.nid), 'label'] = 1
    print("After balance:", nodes_df_tmp['label'].value_counts())
    return nodes_df_tmp


def simtransferL1(nodes_df_tmp, node_pro_time, Adj):
    '''
    只计算相似度，只根据相似度的，余弦相似

    Parameters
    ----------
    nodes_df_tmp : TYPE 需要做标签传播的dataframe
        DESCRIPTION.
    node_pro_time : TYPE  所有节点的产生时间
        DESCRIPTION.
    Adj : TYPE       对应的节点间的关系图
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    res_ids_unkown = nodes_df_tmp[nodes_df_tmp['label'] == 2]  # .drop(columns=['index', 'nid', 'label', 'time'])
    res_ids_other = nodes_df_tmp[
        (nodes_df_tmp['label'] == 0) | (nodes_df_tmp['label'] == 1)]  # .drop(columns=['index', 'nid', 'label', 'time'])
    res_ids_unkown_np = np.matrix(res_ids_unkown[feat].values)
    res_ids_other_np = np.matrix(res_ids_other[feat].values)
    res_ids_other_label = np.matrix(nodes_df_tmp[(nodes_df_tmp['label'] == 0) | (nodes_df_tmp['label'] == 1)]['label'])
    from sklearn.metrics.pairwise import cosine_similarity

    # 1.求余弦相似度
    res_ids = cosine_similarity(res_ids_unkown_np, res_ids_other_np)  # K,M
    res_ids_max, res_ids_min = res_ids.max(axis=0), res_ids.min(axis=0)
    res_ids = (res_ids - res_ids_min) / (res_ids_max - res_ids_min)

    res_ids_label = res_ids * res_ids_other_label.T  # K,1的矩阵
    resmean = res_ids_label.mean()

    # print("res_ids_label", res_ids_label)
    def change1(x):
        if x > 0.5:
            return 1
        else:
            return 0

    res_ids_unkown['label'] = res_ids_label  # 非0，1
    # 二值化
    res_ids_unkown['label'] = res_ids_unkown['label'].apply(change1)
    global count
    count = 0

    def change2(x):
        # global count
        # global num_balance
        # if count>num_balance:
        #             return x
        if x['label'] == 2:
            tmp = res_ids_unkown[res_ids_unkown['nid'] == x['nid']]
            if tmp['label'].values[0] == 1:
                # count = count + 1
                x['label'] = 1
            # else:#给打的标记是正常
            #     x['label'] = 0
            # count = count+1
        # count=0
        return x

    nodes_df_tmp1 = nodes_df_tmp.apply(lambda x: change2(x), axis=1)

    return nodes_df_tmp1