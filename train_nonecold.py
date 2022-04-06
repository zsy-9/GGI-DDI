from DD_pre import DD_Pre
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset, download_url
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
random.seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path1 = '/public/home/yuhui/topk/result_1.txt'
atom_feature = 45
inter_embedding = np.random.rand(1318, 45)
file_handle1 = open(file_path1, 'a')
file_handle1.write('\n')
text = ' '.join(str(i) for i in inter_embedding[58])
file_handle1.write(text)
file_handle1.write('\n')
text = ' '.join(str(i) for i in inter_embedding[59])
file_handle1.write(text)
file_handle1.write('\n')
text = ' '.join(str(i) for i in inter_embedding[60])
file_handle1.write(text)
file_handle1.write('\n')
text = ' '.join(str(i) for i in inter_embedding[102])
file_handle1.write(text)
file_handle1.write('\n')
text = ' '.join(str(i) for i in inter_embedding[103])
file_handle1.write(text)
file_handle1.write('\n')
file_handle1.close()
inter_embedding = torch.FloatTensor(inter_embedding)


def binary_evaluation_result(label_true, score_predict):
    precision, recall, _ = metrics.precision_recall_curve(label_true, score_predict)
    pr_auc_score = metrics.auc(recall, precision)
    return pr_auc_score


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t, Inter, Label):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.Inter = Inter
        self.Label = Label

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    # def __cat_dim__(self, key, value):
    #     if key == 'Inter':
    #         return 0
    #     else:
    #         return super().__cat_dim__(key, value)


def DDI_List(dataloader):
    with open(f'/public/home/yuhui/topk' + dataloader, 'rb') as f:
        drugdata = pickle.load(f)
        Drug1_X = []
        Drug2_X = []
        Drug1_E = []
        Drug2_E = []
        D_Inter = []
        D_Label = []
        Data_list = []
        for i in range(0, len(drugdata)):
            item = drugdata[i]
            D_drug1 = item['drug_1']
            D_drug2 = item['drug_2']
            #Drug1_X.append(D_drug1['x'])
            #Drug1_E.append(D_drug1['edge_index'])
            #Drug2_X.append(D_drug2['x'])
            #Drug2_E.append(D_drug2['edge_index'])
            Interact = inter_embedding[item['Inter']]
            Interact = torch.unsqueeze(Interact, 0)
            #D_Inter.append(Interact)
            Label = item['Label']
            #D_Label.append(Label)
            Data_list.append(PairData(D_drug1['edge_index'], D_drug1['x'], D_drug2['edge_index'], D_drug2['x'], Interact, Label))
    return Data_list


train_set = DDI_List('/train_7.pkl')
test_set = DDI_List('/test_7.pkl')

loader = DataLoader(train_set, batch_size=256, follow_batch=['x_s', 'x_t'])
test_loader = DataLoader(test_set, batch_size=256, follow_batch=['x_s', 'x_t'])

from torch import optim
from DD_pre import DD_Pre
import time

model = DD_Pre(45, 0.6, 0.6).to(device)

file_handle1 = open(file_path1, 'a')
file_handle1.write('\n')
file_handle1.write('0.6*0.6')
file_handle1.close()

# 优化函数
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
# 损失函数
loss = torch.nn.MSELoss(reduction='mean')


# 测试
def test_DDI(test_loader, model):
    model.eval()
    y_pred = []
    y_label = []
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch.edge_index_s = Variable(batch.edge_index_s)
        batch.x_s = Variable(batch.x_s)
        batch.x_s_batch = Variable(batch.x_s_batch)
        batch.edge_index_t = Variable(batch.edge_index_t)
        batch.x_t = Variable(batch.x_t)
        batch.x_t_batch = Variable(batch.x_t_batch)
        batch.Inter = Variable(batch.Inter)
        predictions = model(batch.edge_index_s, batch.x_s, batch.x_s_batch, batch.edge_index_t, batch.x_t,
                            batch.x_t_batch, batch.Inter)
        Label = batch.Label
        predictions = predictions.detach().cpu().numpy()
        Label = Label.detach().cpu().numpy()
        y_label = y_label + Label.flatten().tolist()
        y_pred = y_pred + predictions.flatten().tolist()
    y_pred1 = np.array(y_pred)
    y_label1 = np.array(y_label)
    roc_test_AUC, roc_test_Pre, roc_test_AUPR, roc_test_ACC = roc_auc_score(y_label, y_pred), average_precision_score(
        y_label1, y_pred1.round(), average='micro'), binary_evaluation_result(y_label1,
                                                                              y_pred1.round()), accuracy_score(y_label1,
                                                                                                               y_pred1.round())
    return roc_test_AUC, roc_test_Pre, roc_test_AUPR, roc_test_ACC


# 训练
loss_history = []
t_total = time.time()
epochs = 45
for epoch in range(epochs):
    t = time.time()
    y_pred_train = []
    y_label_train = []
    for i, batch in enumerate(loader):
        model.train(True)
        batch = batch.to(device)
        batch.edge_index_s = Variable(batch.edge_index_s)
        batch.x_s = Variable(batch.x_s)
        batch.x_s_batch = Variable(batch.x_s_batch)
        batch.edge_index_t = Variable(batch.edge_index_t)
        batch.x_t = Variable(batch.x_t)
        batch.x_t_batch = Variable(batch.x_t_batch)
        batch.Inter = Variable(batch.Inter)
        predictions = model(batch.edge_index_s, batch.x_s, batch.x_s_batch, batch.edge_index_t, batch.x_t,
                            batch.x_t_batch, batch.Inter)
        Label = batch.Label.float()
        # Label = torch.FloatTensor(Label)
        loss1 = loss(predictions, Label)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        predictions = predictions.detach().cpu().numpy()
        Label = Label.detach().cpu().numpy()
        y_label_train = y_label_train + Label.flatten().tolist()
        y_pred_train = y_pred_train + predictions.flatten().tolist()
    roc_train = roc_auc_score(y_label_train, y_pred_train)
    print(roc_train)
    file_handle1 = open(file_path1, 'a')
    file_handle1.write('\n')
    file_handle1.write('roc_train:%f ' % roc_train)
    file_handle1.close()
    AUC_T, Pre_T, AUPR_T, ACC_T = test_DDI(test_loader, model)
    file_handle1 = open(file_path1, 'a')
    file_handle1.write('\n')
    file_handle1.write('AUC_test:%f ' % AUC_T)
    file_handle1.write('Pre_test:%f ' % Pre_T)
    file_handle1.write('AUPR_test:%f ' % AUPR_T)
    file_handle1.write('ACC_test:%f ' % ACC_T)
    file_handle1.write('\n')
    file_handle1.close()
torch.save(model.state_dict(), '/public/home/yuhui/topk/top_model_2.pt')





