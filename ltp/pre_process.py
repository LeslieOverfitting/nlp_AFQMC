import json
from gensim.models import word2vec
import pyltp as pp


class DataGenerator:
    def __init__(self):
        self.path = '../data/'
        self.unprocessed_data = []
        self.attribute = ['dis', 'bod', 'sym', 'dec', 'fre', 'ite']
        self.dics = {}
        self.processed_data = []
        self.segmentor = pp.Segmentor()
        self.segmentor.load("ltp_data_v3.4.0/cws.model")  # 加载分词模型

    def read_data(self, file_name):
        with open(self.path+file_name, 'r') as f:
            for i in f.readlines():
                self.unprocessed_data.append(json.loads(i))


    def sentence_process(self):
        for i in self.unprocessed_data:
            self.processed_data.append(self.segmentor.segment(i['sentence1']))
            self.processed_data.append(self.segmentor.segment(i['sentence2']))

    def set_word2vec(self):
        with open('../data/w2v.txt', 'a') as f:
            for word in self.processed_data:
                f.write(' '.join(word)+'\n')
        sentences = word2vec.Text8Corpus('data/w2v.txt')
        model = word2vec.Word2Vec(sentences, size=100, min_count=1)
        model.save('../data/wm')

    def return_data(self, file_name):
        data = []
        with open(self.path+file_name, 'r') as f:
            for i in f.readlines():
                data.append(json.loads(i))
        return data


if __name__ == '__main__':
    dg = DataGenerator()
    dg.read_data('train.json')
    dg.read_data('dev.json')
    dg.read_data('test.json')
    dg.sentence_process()
    dg.set_word2vec()

