import pyltp as pp
import gensim
import json
import numpy as np
from scipy.linalg import norm
from sklearn import metrics


class LTPGenerator:
    def __init__(self):
        self.path = 'ltp_data_v3.4.0/' # 下载地址 https://ltp.ai/download.html 3.4.0
        self.segmentor = pp.Segmentor()
        self.segmentor.load(self.path + "cws.model") # 加载分词模型

        self.postagger = pp.Postagger()
        self.postagger.load(self.path + "pos.model")  # 加载词性标注模型

        self.recognizer = pp.NamedEntityRecognizer()
        self.recognizer.load(self.path + "ner.model") # 加载命名实体识别模型

        self.parser = pp.Parser()
        self.parser.load(self.path + "parser.model") # 加载依存句法分析模型

        self.labeller = pp.SementicRoleLabeller()
        self.labeller.load(self.path + "pisrl.model") # 加载语义角色标注模型

    def analysis(self, text):
        dics = {}
        # 分词
        words = self.segmentor.segment(text)
        # print('分词')
        # print(' '.join(words))

        # 词性标注
        postags = self.postagger.postag(words)
        # print('词性标注')
        # print(' '.join(postags))
        for i in postags:
            if i in dics:
                dics[i]+=1
            else:
                dics[i]=1

        # 句法分析
        arcs = self.parser.parse(words, postags)
        # print('句法分析')
        # print("\t".join(
        #     "%d:%s" % (arc.head, arc.relation) for arc in arcs))

        return words,dics,arcs,postags

    # 释放模型
    def release_model(self):
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()
        self.parser.release()
        self.labeller.release()

def getSim():
    sub1 = np.zeros(100)
    for i in words:
        sub1 += model.wv[i]
    sub1 = sub1/len(words)
    sub2 = np.zeros(100)
    for i in words2:
        sub2 += model.wv[i]
    sub2 = sub2/len(words2)
    return np.dot(sub1, sub2) / (norm(sub1) * norm(sub2))


if __name__=='__main__':
    with open('../data/IS.txt','r') as f:
        IS = f.read().split(' ')
    ltp = LTPGenerator()
    model = gensim.models.Word2Vec.load('data/wm')
    with open('../data/dev.json', 'r') as f:
        tests = []
        for i in f.readlines():
            tests.append(json.loads(i))
    acc = 0
    num = 0
    loss = 0
    predict = []
    target = []
    for i in tests[:]:
        text = i['sentence1']
        words, dics, arcs, pos = ltp.analysis(text)
        text2 = i['sentence2']
        words2, dics2, arcs2, pos2 = ltp.analysis(text2)
        b = getSim()
        count = b
        # print('相似度',count,'label',i['label'])
        num += 1
        if count >= 0.5 and i['label'] == '1':
            acc += 1
        if count < 0.5 and i['label'] == '0':
            acc += 1
        loss+=(b-int(i['label']))**2
        if count >= 0.5:
            predict.append(1)
        else:
            predict.append(0)
        target.append(int(i['label']))

    ltp.release_model()
    print('准确率',acc/num)
    print('损失',loss**0.5)
    print(metrics.f1_score(predict,target,average='weighted'))












