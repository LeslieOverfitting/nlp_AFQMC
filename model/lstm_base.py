import torch 
import torch.nn as nn 
import torch.nn.functional as F
from model.sentenceEncoder import SentenceEncoder
from utils import init_model_weights
class LSTMBase(nn.Module):
    def __init__(self, config, word_emb):
        super(LSTMBase, self).__init__()
        self.hidden_size = config.hidden_size
        self.n_classes = config.n_classes
        self.dropout = config.dropout
        self.hidden_layer = config.hidden_layer
        self.device = config.device
        self.padding_idx = config.padding_idx
        self.emb = nn.Embedding.from_pretrained(torch.tensor(word_emb))
        self.emb_dim = self.emb.embedding_dim
        # lstm
        self.encoder_layer = SentenceEncoder(self.emb_dim, 
                                             self.hidden_size, 
                                             num_layers=self.hidden_layer, 
                                             bias=True, 
                                             dropout=self.dropout)
        self.predict_fc = nn.Sequential(nn.Dropout(p=self.dropout),
                                            nn.Linear(2 * 2 * self.hidden_size, self.hidden_size),
                                            nn.Tanh(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size, self.n_classes)
                                            )
        self.apply(init_model_weights)
    def forward(self, sentences1, sentences2):
        '''
            sentences1: [batch_size, max_len]
            sentences2: [batch_size, max_len]
        '''
        embedded_sentences1 = self.emb(sentences1)# [batch_size, max_len, dim]
        embedded_sentences2 = self.emb(sentences2)# [batch_size, max_len, dim]
        sentences1_mask = (sentences1 != self.padding_idx).long().to(self.device)# [batch_size, max_len, dim]
        sentences2_mask = (sentences2 != self.padding_idx).long().to(self.device)
        sentences1_len = torch.sum(sentences1_mask, dim=-1).view(-1)# [batch_size]
        sentences2_len = torch.sum(sentences2_mask, dim=-1).view(-1)# [batch_size]
        encoded_sentences1 = self.encoder_layer(embedded_sentences1, sentences1_len)
        encoded_sentences2 = self.encoder_layer(embedded_sentences2, sentences1_len)
        
        sentences1_len = sentences1_len.view(-1, 1) # [batch_size, 1]
        sentences2_len = sentences2_len.view(-1, 1) # [batch_size, 1]

        sentences1_mean = torch.sum(embedded_sentences1, dim=1) / sentences1_len # [batch_size, dim] / [batch_size, 1]
        sentences2_mean = torch.sum(embedded_sentences2, dim=1) / sentences2_len
        # [s1_mean; s2_mean; s1_mean - s2_mean]
        cat = torch.cat([sentences1_mean, sentences2_mean, sentences1_mean - sentences2_mean], dim=1)
        return self.predict_fc(cat)