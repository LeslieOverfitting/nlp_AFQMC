import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import sort_by_seq_lens, get_mask, init_model_weights, replace_masked
from model.sentenceEncoder import SentenceEncoder

class SSE(nn.Module):

    def __init__(self, config, word_emb, hidden_size=[256, 512, 1024]):
        super(SSE, self).__init__()
        self.n_vocab = config.n_vocab
        self.hidden_size = hidden_size
        self.emb_dim = config.emb_dim
        self.word_emb = word_emb
        self.n_classes = config.n_classes
        self.padding_idx = config.padding_idx
        self.dropout = config.dropout
        self.hidden_layer = config.hidden_layer
        self.device = config.device
        self.emb = nn.Embedding.from_pretrained(torch.tensor(word_emb))
        
        self.shortcut_lstm1 = SentenceEncoder(input_size=self.emb_dim, 
                                              hidden_size=self.hidden_size[0],  
                                              num_layers=1, 
                                              bias=True, 
                                              dropout=self.dropout)
        self.shortcut_lstm2 = SentenceEncoder(input_size=self.emb_dim + 2 * self.hidden_size[0], 
                                              hidden_size=self.hidden_size[1],  
                                              num_layers=1, 
                                              bias=True, 
                                              dropout=self.dropout)
        self.shortcut_lstm3 = SentenceEncoder(input_size=self.emb_dim + 2 * self.hidden_size[0] + 2 * self.hidden_size[1], 
                                              hidden_size=self.hidden_size[2],  
                                              num_layers=1, 
                                              bias=True, 
                                              dropout=self.dropout)
        self.mlp = nn.Sequential(nn.Dropout(p=self.dropout),
                                nn.Linear(self.hidden_size[2] * 2 * 4, self.n_classes))
        self.apply(init_model_weights)

    def forward(self, sentences1, sentences2):
        """
            sentence1 [batch, max_len]
            sentence2 [batch, max_len]
        """
        # get mask
        sentences1_mask = (sentences1 != self.padding_idx).long().to(self.device) # [batch_size, max_len]
        sentences2_mask = (sentences2 != self.padding_idx).long().to(self.device) # [batch_size, max_len]
        # input encoding
        sentences1_emb = self.emb(sentences1) # [batch_size, max_len, dim]
        sentences2_emb = self.emb(sentences2) # [batch_size, max_len, dim]
        sentences1_len = torch.sum(sentences1_mask, dim=-1).view(-1)# [batch_size]
        sentences2_len = torch.sum(sentences2_mask, dim=-1).view(-1)# [batch_size]
        # shortcut lstm1
        q1_lstm1_outputs = self.shortcut_lstm1(sentences1_emb, sentences1_len) # [batch_size, max_len_q1, hidden_size[0]]
        q2_lstm1_outputs = self.shortcut_lstm1(sentences2_emb, sentences2_len) # [batch_size, max_len_q2, hidden_size[0]]
       
        # shortcut lstm2
        q1_lstm2_inputs = torch.cat([sentences1_emb, q1_lstm1_outputs], dim=-1)# [batch_size, max_len_q1, emb_dim + 2*hidden_size[0]]
        q2_lstm2_inputs = torch.cat([sentences2_emb, q2_lstm1_outputs], dim=-1)
        q1_lstm2_outputs = self.shortcut_lstm2(q1_lstm2_inputs, sentences1_len) # [batch_size, max_len_q1, 2*hidden_size[1]]
        q2_lstm2_outputs = self.shortcut_lstm2(q2_lstm2_inputs, sentences2_len) # [batch_size, max_len_q2, 2*hidden_size[1]]
       
        # shortcut lstm3
        q1_lstm3_inputs = torch.cat([sentences1_emb, q1_lstm1_outputs, q1_lstm2_outputs], dim=-1)# [batch_size, max_len_q1, emb_dim + 2*hidden_size[0] + 2*hidden_size[1]]
        q2_lstm3_inputs = torch.cat([sentences2_emb, q2_lstm1_outputs, q2_lstm2_outputs], dim=-1)
        q1_encoded = self.shortcut_lstm3(q1_lstm3_inputs, sentences1_len) # [batch_size, max_len_q1, hidden_size[2]*2]
        q2_encoded = self.shortcut_lstm3(q2_lstm3_inputs, sentences2_len) # [batch_size, max_len_q2, hidden_size[2]*2]

        # max pool
        s1_encoded = replace_masked(q1_encoded, sentences1_mask, -1e7) # [batch_size, seq1_len, hidden_size[2]*2]]
        s2_encoded = replace_masked(q2_encoded, sentences2_mask, -1e7) # [batch_size, seq1_len, hidden_size[2]*2]]
        max_encoded_q1 = torch.max(s1_encoded, dim=1)[0] # [batch_size, hidden_size[2] * 2]
        max_encoded_q2 = torch.max(s2_encoded, dim=1)[0]

        # combine m = [vp; vh; jvp − vhj ; vp ⊗ vh]
        m = torch.cat([max_encoded_q1, max_encoded_q2, torch.abs(max_encoded_q1 - max_encoded_q2), max_encoded_q1 * max_encoded_q2], dim=1)
        predict = self.mlp(m)
        return predict

