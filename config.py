import torch
class Config:
    def __init__(self):
        # 超参数
        self.n_classes = 2
        self.dropout = 0.2
        self.hidden_layer = 2
        self.padding_idx = 0
        self.batch_size = 32
        self.hidden_size = 200

        # file path
        self.emb_path = 'pretrainEmb/sgns.zhihu.bigram-char'
        self.train_data_path = 'data/train.csv'
        self.dev_data_path = 'data/dev.csv'
        self.test_data_path = 'data/test.csv'
        self.model_save_path = 'saveModel/lstm_base.pt'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')