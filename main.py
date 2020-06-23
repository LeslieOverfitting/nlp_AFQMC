 
import pandas as pd
import torch
import json
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from sentence_dataSet import SentenceDataSet
from data_processor import DataProcessor
from executor import Executor
from model.lstm_base import LSTMBase
from model.esim import ESIM
from model.sse import SSE
from model.inferSent import InferSent
from config import Config
from utils import *


MODEL_CLASSES = {
    'lstm_base': LSTMBase,
    'esim': ESIM,
    'sse': SSE,
    'inferSent': InferSent
}

def train(config, model_name, data_processor, executor):
    # 加载数据
    train_df = pd.read_csv(config.train_data_path)
    dev_df = pd.read_csv(config.dev_data_path)

    # 生成训练数据样本
    train_data_set = data_processor.get_dataset(train_df)
    dev_data_set = data_processor.get_dataset(dev_df)
    train_loader = DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data_set, batch_size=config.batch_size, shuffle=True)

    # 加载模型
    model = MODEL_CLASSES[model_name](config, data_processor.emb_matrix)
    print(model)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(str(total_trainable_params), 'parameters is trainable.')
    model.to(config.device)
    es = EarlyStopping()
    for i in range(config.epoch_num):
        print('Epoch:  ', i + 1)
        executor.train_model(train_loader, model)
        dev_acc, dev_loss, report, confusion, f1_score = executor.evaluate_model(dev_loader, model)
        print_ans(dev_acc, dev_loss, report, confusion)
        es(f1_score, model, config.model_save_path)
        if es.early_stop:
          print("Early stopping")
          break
    print('best model')
    model.load_state_dict(torch.load(config.model_save_path))
    dev_acc, dev_loss, report, confusion, f1_score = executor.evaluate_model(dev_loader, model)
    print_ans(dev_acc, dev_loss, report, confusion)

def inference(config, model_name, data_processor, executor):
    # 加载数据
    test_df = pd.read_csv(config.train_data_path)
    # 生成测试数据样本
    test_data_set = data_processor.get_dataset(test_df, is_train=False)
    test_loader = DataLoader(test_data_set, batch_size=config.batch_size, shuffle=True)
    # 加载模型
    model = MODEL_CLASSES[model_name](config, data_processor.emb_matrix)
    model.load_state_dict(torch.load(config.model_save_path))
    device = config.device
    model.to(device)
    # 预测结果
    predicts_all = executor.inference(test_loader, model)
    # 保存标签结果
    output_submit_file = config.predict_output_path
    with open(output_submit_file, "w") as writer:
        for i, pred in enumerate(predicts_all):
            json_d = {}
            json_d['id'] = i
            json_d['label'] = str(pred)
            writer.write(json.dumps(json_d) + '\n')
    print('inference over')

if __name__ =='__main__':
    config = Config()
    data_processor = DataProcessor(config)
    executor = Executor(config)
    model_name = 'sse'
    config.model_save_path = f'saveModel/{model_name}.pt'
    # 训练模型
    train(config, data_processor, executor)
    # 预测结果
    inference(config, model_name, data_processor, executor)
