import torch
class Config:
    def __init__(self):
        # 超参数
        self.n_classes = 2
        self.emb_dim = 300
        self.n_vocab = 0
        self.dropout = 0.2
        self.hidden_layer = 2
        self.padding_idx = 0
        self.batch_size = 64
        self.hidden_size = 200
        self.max_len = 35
        self.epoch_num = 30
        self.learn_rate = 3e-4 # 当模型为 bert 时候修改为 2e-5，否则预测诡异
        # file path
        self.emb_path = 'pretrainEmb/sgns.financial.bigram-char'
        # 文件目录
        self.train_data_path = 'data/train.csv'
        self.dev_data_path = 'data/dev.csv'
        self.test_data_path = 'data/test.csv'
        self.model_save_path = 'saveModel/lstm_base.pt'
        self.predict_output_path = 'data/submission.json'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = 'bert_pretrained/'
        self.bert_model_path = self.pretrained_path + 'bert_base_chinese/bert-base-chinese-pytorch_model.bin'
        self.bert_config_path = self.pretrained_path + 'bert_base_chinese/bert-base-chinese-config.json'
        self.bert_vocab_path = self.pretrained_path + 'bert_base_chinese/bert-base-chinese-vocab.txt'