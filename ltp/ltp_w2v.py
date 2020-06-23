import pyltp as pp
import gensim
import json
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

def getSim2():
    post_common = 0
    post_all = 0
    for tag in tags:
        if tag in dics and tag in dics2:
            post_common += min(dics2[tag], dics[tag])
        if tag in dics:
            post_all += dics[tag]
        if tag in dics2:
            post_all += dics2[tag]
    b2 = post_common*2 / post_all
    return b2

def getSim1():
    r1 = 0
    r2 = 0
    for i in IS:
        if i in text:
            r1 = 1
            break
    for i in IS:
        if i in text2:
            r2 = 1
            break
    if r1==r2:
        return 1
    else:
        return 0.1

def getSim3():
    d1 = None
    d2 = None
    for i in range(len(pos)):
        if pos[i]=='d':
            d1 = i
    for i in range(len(pos2)):
        if pos2[i]=='d':
            d2 = i
    if d1 and d2:
        if model.wv.similarity(words[d1],words2[d2])>0.1:
            return 1
        else:
            return -1
    else:
        return 1

def getSim4():
    sub1 = None
    ver1 = None
    obj1 = None
    index = -1
    for i in range(len(pos)):
        if pos[i] in ['n','nd','nh','ni','nl','ns','r','nz']:
            sub1 = words[i]
            index = i
            break
    for i in range(index+1,len(pos)):
        if pos[i]=='v':
            ver1 = words[i]
            break
    for i in range(index+1,len(pos)):
        if pos[i] in ['n','nd','nh','ni','nl','ns','r','nz']:
            obj1 = words[i]
            break
    sub2 = None
    ver2 = None
    obj2 = None
    index2 = -1
    for i in range(len(pos2)):
        if pos2[i] in ['n','nd','nh','ni','nl','ns','r','nz']:
            sub2 = words2[i]
            index2 = i
            break
    for i in range(index2+1,len(pos2)):
        if pos2[i]=='v':
            ver2 = words2[i]
            break
    for i in range(index2+1,len(pos2)):
        if pos2[i] in ['n','nd','nh','ni','nl','ns','r','nz']:
            obj2 = words2[i]
            break
    res = 0
    if sub1 and sub2:
        res += 0.3*model.wv.similarity(sub1,sub2)

    if ver1 and ver2:
        res += 0.5*model.wv.similarity(ver1,ver2)

    if obj1 and obj2:
        res += 0.2*model.wv.similarity(obj1,obj2)

    return res


if __name__=='__main__':
    tags = ['a','b','c','d','e','g','h','i','j','k','m','n','nd','nh','ni','nl','ns','nt','nz','o','p','q','r','u','v','wp','ws','x','z']
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
        b1 = getSim1()
        b2 = getSim2()
        b3 = getSim3()
        b4 = getSim4()
        count = b1 * b2 * b3 * b4
        # print('相似度',count,'label',i['label'])
        num+=1
        if count>=0.5 and i['label']=='1':
            acc+=1
        if count<0.5 and i['label']=='0':
            acc+=1
        loss += (count - int(i['label'])) ** 2
        if count >= 0.5:
            predict.append(1)
        else:
            predict.append(0)
        target.append(int(i['label']))
    ltp.release_model()
    print('准确率',acc/num)
    print('损失', loss ** 0.5)
    print(metrics.f1_score(predict, target, average='weighted'))












