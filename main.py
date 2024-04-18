import sys
import numpy as np
import torch.nn as nn
import torch

from Mmodel import Mmodel
from MCPDataset import MCPDataset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve,recall_score

# %%超参数
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 0.001
LOG_INTERVAL = 10
modeling = Mmodel
cuda_name = "cuda:0"

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data["graphImg"] = data["graphImg"].float().to(device).permute(0, 3, 1, 2)
        data["ecfp"] = data["ecfp"].float().to(device)
        data["hash"] = data["hash"].float().to(device)
        data["list_of_node"] = data["list_of_node"].float().to(device)
        data["list_of_edge"] = data["list_of_edge"].float().to(device)
        data["label"] = data["label"].float().to(device)
        optimizer.zero_grad()
        preds = model(data["graphImg"],data["ecfp"],data["hash"],
                       data["list_of_node"],data["list_of_edge"])
        # preds = output.cpu()
        loss = loss_fn(preds,data["label"])
        loss.backward()
        optimizer.step()
        return preds,data["label"]

def predicting(model, device, loader):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data["graphImg"] = data["graphImg"].float().to(device).permute(0, 3, 1, 2)
            data["ecfp"] = data["ecfp"].float().to(device)
            data["hash"] = data["hash"].float().to(device)
            data["list_of_node"] = data["list_of_node"].float().to(device)
            data["list_of_edge"] = data["list_of_edge"].float().to(device)
            data["label"] = data["label"].float().to(device)
            output = model(data["graphImg"],data["ecfp"],data["hash"],
                       data["list_of_node"],data["list_of_edge"])
            preds = output.cpu()
    return preds,data["label"]

if __name__ == '__main__':
    n_train = len(MCPDataset())
    split = n_train // 5
    indices = np.random.choice(range(n_train), size=n_train, replace=False)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_loader = DataLoader(MCPDataset(), sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
    test_loader = DataLoader(MCPDataset(), sampler=test_sampler, batch_size=TEST_BATCH_SIZE)

    # %%训练模型
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    AUClist=[]
    ACClist=[]
    for epoch in range(NUM_EPOCHS):
        GT, GP = train(model, device, train_loader, optimizer)
        G, P = predicting(model, device, test_loader)
        G =G.cpu()
        P=P.cpu()
        fpr, tpr, thresholds = roc_curve(P, G)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_new = (G >= optimal_threshold).to(torch.int)
        roc_auc = roc_auc_score(P, y_pred_new)
        ACC = accuracy_score(P, y_pred_new)
        recall = recall_score(P, y_pred_new)
        tn, fp, fn, tp = confusion_matrix(P, y_pred_new).ravel()  
        specificity = tn / (tn + fp)
        AUClist.append(roc_auc)
        AUCbest = max(AUClist)
        ACClist.append(ACC)
        ACCbest = max(ACClist)
        if epoch % LOG_INTERVAL == 0:
            print("This is:",epoch,"epoch,bestAUC:",AUCbest,"bestACC:",ACCbest)
            # print("AUC:",roc_auc,"ACC:", ACC,"se:", recall,"sp:", specificity)



