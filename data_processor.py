from tokenizer import Tokenizer
import torch
from sentence_dataSet import SentenceDataSet
from utils import *
import jieba
import collections
class DataProcessor:
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb, dict_len, emb_dim = self.get_emb(config.emb_path)
        self.tokenizer = Tokenizer(emb.keys())
        self.dict_len = dict_len + 2 # pad unk
        emb_matrix = self.get_emb_matrix(emb, self.tokenizer, dict_len, emb_dim) # key index value word_emb
        self.emb_matrix = torch.tensor(emb_matrix)
    def get_dataset(self, df, is_train=True):
        # 将语句字符串 转化为 index
        sentence1_indexs = list(map(self.convert_sentence_to_index, df['sentence1'].astype(str)))
        sentence2_indexs = list(map(self.convert_sentence_to_index, df['sentence2'].astype(str)))
        data_set = None
        if is_train:
            data_set = SentenceDataSet(sentence1_indexs, sentence2_indexs, df['label'].astype(int))
        else:
            data_set = SentenceDataSet(sentence1_indexs, sentence2_indexs, None)
        return data_set

    def get_emb(self, emb_path):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        with open(emb_path, 'r', encoding='utf-8') as emb_file:
            dict_len, emb_dim = emb_file.readline().rstrip().split()
            print('dict_len: ', dict_len)
            print('emb dim:', emb_dim)
            dict_len, emb_dim = int(dict_len), int(emb_dim)
            emb = collections.OrderedDict(get_coefs(line.rstrip().split()) for line in emb_file.readlines())
        return emb, dict_len, emb_dim

    
    def get_emb_matrix(self, emb, tokenizer, dict_len, emb_dim):
        '''
            input: 
                    emb 词向量矩阵 list
                    dict_len 词数量
                    emb_dim 词向量维度
            return:  
                    emb_matrix: key index value word_emb
        '''
        emb_matrix = np.random.randn((2 + dict_len, emb_dim), dtype='float32')
        for word, idx in tokenizer.vocab.item():
            emb_vector = emb.get(word)
            if emb_vector is not None:
                emb_matrix[idx] = emb_vector
        return emb_matrix
    
    def convert_sentence_to_index(self, sentence):
        tokens_list = jieba.cut(sentence) # 利用结巴分词对语句进行切词
        ids_index = self.tokenizer.tokens_to_id(tokens_list)# 将字词转化为对应的词向量下标
        # padding 对未达到 max_len 的序列进行填充
        ids_index = ids_index[:self.config.max_len] +  max(self.config.max_len - len(ids_index), 0) * [0]
        return ids_index
    
    