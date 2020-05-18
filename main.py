from utils import *
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from sentence_dataSet import SentenceDataSet
from data_processor import DataProcessor
from executor import Executor
from model.lstm_base import LSTMBase
from config import Config
import pandas as pd
import torch

def train(config, data_processor, executor):
    # 加载数据
    train_df = pd.read_csv(config.train_data_path)
    dev_df = pd.read_csv(config.dev_data_path)

    # 生成训练数据样本
    train_data_set = data_processor.get_dataset(train_df)
    dev_data_set = data_processor.get_dataset(train_df)
    train_loader = DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data_set, batch_size=config.batch_size, shuffle=True)

    # 加载模型
    model = LSTMBase(config, data_processor.emb_matrix)
    print(model)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(str(total_trainable_params), 'parameters is trainable.')

    dev_best_loss = float('inf')
    for i in range(config.epcoh_num):
        print('Epcoh:  ', i)
        executor.train_model(train_loader, model)
        dev_acc, dev_loss, report, confusion = executor.evaluate_model(dev_loader, model)
        print_ans(dev_acc, dev_loss, report, confusion)
        if dev_loss < dev_best_loss: # 保存最好的模型
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.model_save_path)

def test(config,  data_processor, executor):
    # 加载数据
    test_df = pd.read_csv(config.train_data_path)
    # 生成测试数据样本
    test_data_set = data_processor.get_dataset(test_df, is_train=False)
    test_loader = DataLoader(test_data_set, batch_size=config.batch_size, shuffle=True)
    # 加载模型
    model = LSTMBase(config, data_processor.emb_matrix)
    model.load_state_dict(torch.load(config.model_save_path))
    #model
    test_acc, test_loss, report, confusion = executor.evaluate_model(test_loader, model)
    print_ans(test_acc, test_loss, report, confusion)
    


if __name__ == '__main__':
    config = Config()
    data_processor = DataProcessor(config)
    executor = Executor(config)
