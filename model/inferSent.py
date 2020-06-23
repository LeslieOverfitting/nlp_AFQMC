import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sentenceEncoder import SentenceEncoder
from utils import sort_by_seq_lens, get_mask, replace_masked, init_model_weights

class InferSent(nn.Module):

    def __init__(self, config, word_emb):
        super(InferSent, self).__init__()
        self.emb_dim = config.emb_dim
        self.word_emb = word_emb
        self.hidden_size = config.hidden_size
        self.hidden_layer = config.hidden_layer
        self.n_classes = config.n_classes
        self.dropout = config.dropout
        self.padding_idx = config.padding_idx
        self.device = config.device
        self.emb = nn.Embedding.from_pretrained(torch.tensor(word_emb))
        # 编码层
        self.encoder_layer = SentenceEncoder(input_size=self.emb_dim, 
                                             hidden_size=self.hidden_size, 
                                             num_layers=self.hidden_layer, 
                                             bias=True, 
                                             dropout=self.dropout)
        # 预测
        self.mlp = nn.Sequential(nn.Dropout(p=self.dropout),
                                nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                nn.Tanh(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(self.hidden_size, self.n_classes)
                                )
        self.apply(init_model_weights)

    def forward(self, sentences1, sentences2):
        """
            input:
                sentences1: [batch_size, seq1_len]
                sentences2: [batch_size, seq2_len]
        """
        # get mask
        sentences1_mask = (sentences1 != self.padding_idx).long().to(self.device) # [batch_size, max_len]
        sentences2_mask = (sentences2 != self.padding_idx).long().to(self.device) # [batch_size, max_len]

        # input encoding
        sentences1_emb = self.emb(sentences1) # [batch_size, max_len, dim]
        sentences2_emb = self.emb(sentences2) # [batch_size, max_len, dim]
        sentences1_len = torch.sum(sentences1_mask, dim=-1).view(-1)# [batch_size]
        sentences2_len = torch.sum(sentences2_mask, dim=-1).view(-1)# [batch_size]
        
        # encoder
        s1_encoded = self.encoder_layer(sentences1_emb, sentences1_len) # [batch_size, max_len_q1, 2 * dim]
        s2_encoded = self.encoder_layer(sentences2_emb, sentences2_len) # [batch_size, max_len_q2, 2 * dim]

        # max pool
        s1_encoded = replace_masked(s1_encoded, sentences1_mask, -1e7) # [batch_size, seq1_len, hidden_size * 2]
        s2_encoded = replace_masked(s2_encoded, sentences2_mask, -1e7) # [batch_size, seq1_len, hidden_size * 2]
        u = torch.max(s1_encoded, 1)[0]
        v = torch.max(s2_encoded, 1)[0]
        # (u, v, |u − v|, u ∗ v)
        merge = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        logits = self.mlp(merge)
        return logits
