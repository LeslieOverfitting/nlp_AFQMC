import torch 
import numpy as np
from torch.utils.data import Dataset

class SentenceDataSet(Dataset):
    def __init__(self, sentences1, sentences2, labels=None):
        super().__init__()
        self.sentences1 = np.asarray(sentences1)
        self.sentences2 = np.asarray(sentences2)
        if labels is not None:
            self.labels = np.asarray(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.sentences1)
    
    def __getitem__(self, index):
        if self.labels is None:
            return torch.tensor(self.sentences1[index], dtype=torch.long), torch.tensor(self.sentences2[index], dtype=torch.long)
        else:
            return torch.tensor(self.sentences1[index], dtype=torch.long), torch.tensor(self.sentences2[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)

