import torch 
from torch.utils.data import Dataset

class SentenceDataSet(Dataset):
    def __init__(self, sentences1, sentences2, labels=None):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

    def __len__(self):
        return len(self.sentences1)
    
    def __getitem__(self, index):
        if self.labels is None:
            return self.sentences1[index], self.sentences2[index]
        else:
            return self.sentences1[index], self.sentences2[index], self.labels[index]

