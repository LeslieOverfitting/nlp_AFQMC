import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import time
from utils import get_time_diff

class Executor:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def train_model(self, data_loader, model):
        # 训练模型
        start_time = time.time() #获取起始位置
        optimizer = optim.Adam(model.parameters(), lr=self.config.learn_rate, betas=(0.9, 0.999))
        model.train()
        criterion = nn.CrossEntropyLoss()
        total_batch = 0
        for data_batch in data_loader:
            total_batch += 1
            sentences1 = data_batch[0]
            sentences2 = data_batch[1]
            labels = data_batch[2]
            if torch.cuda.is_available(): # 判断设备信息
                sentences1 = data_batch[0].to(self.config.device)
                sentences2 = data_batch[1].to(self.config.device)
                labels = data_batch[2].to(self.config.device)
            optimizer.zero_grad()
            outputs = model(sentences1, sentences2)
            #计算 loss
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                true_label = labels.data.cpu().numpy()
                predict = torch.max(outputs, dim=1)[1].cpu().numpy() #获取预测结果
                train_acc = metrics.accuracy_score(true_label, predict)
                time_diff = get_time_diff(start_time)
                msg = 'Iter:{0:>6} Train loss: {1:>5.3} Train acc:{2:>6.2%} Time:{3}'
                print(msg.format(total_batch, loss.item(), train_acc, time_diff))


    def evaluate_model(self, data_loader, model):
        start_time = time.time()
        model.eval()
        total_loss = 0
        predicts_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data_batch in data_loader:
                sentences1 = data_batch[0]
                sentences2 = data_batch[1]
                labels = data_batch[2]
                if torch.cuda.is_available():
                    sentences1 = data_batch[0].to(self.config.device)
                    sentences2 = data_batch[1].to(self.config.device)
                    labels = data_batch[2].to(self.config.device)
                outputs = model(sentences1, sentences2)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                # 将预测结果写入记录
                predict = torch.max(outputs, dim=1)[1].cpu().numpy()
                labels = labels.data.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predicts_all = np.append(predicts_all, predict)
        #生成评估结果
        acc = metrics.accuracy_score(labels_all, predicts_all)
        report = metrics.classification_report(labels_all, predicts_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all) # 混淆矩阵
        f1_score = metrics.f1_score(labels_all, predicts_all, average='macro')
        return acc, total_loss / len(data_loader), report, confusion, f1_score

    def inference(self, data_loader, model):
        # 预测输出
        predicts_all = np.array([], dtype=int)
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                sentences1 = batch[0]
                sentences2 = batch[1]
                if torch.cuda.is_available():
                    sentences1 = batch[0].to(self.config.device)
                    sentences2 = batch[1].to(self.config.device)
                outputs = model(sentences1, sentences2)
                predict = torch.max(outputs, dim=1)[1].cpu().numpy()
                predicts_all = np.append(predicts_all, predict)

        return predicts_all


