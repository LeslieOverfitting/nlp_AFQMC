import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import sort_by_seq_lens, get_mask, masked_softmax, weighted_sum, replace_masked, init_model_weights
from model.sentenceEncoder import SentenceEncoder

class ESIM(nn.Module):

    def __init__(self, config, word_emb):
        super(ESIM, self).__init__()
        #self.n_vocab = config.n_vocab
        self.hidden_size = config.hidden_size
        self.emb_dim = config.emb_dim
        self.n_classes = config.n_classes
        self.padding_idx = config.padding_idx
        self.dropout = config.dropout
        self.hidden_layer = config.hidden_layer
        self.device = config.device
        self.emb = nn.Embedding.from_pretrained(torch.tensor(word_emb))
        #self.encoder = nn.Embedding(self.n_vocab, self.emb_dim, padding_idx=self.padding_idx)
        #if self.word_emb is not None:
        #    self.encoder.weight.data.copy_(self.word_emb)
        self.encoder_layer = SentenceEncoder(input_size=self.emb_dim, 
                                             hidden_size=self.hidden_size, 
                                             num_layers=self.hidden_layer, 
                                             bias=True, 
                                             dropout=self.dropout)

        self.projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                        nn.ReLU())

        self.composition_layer = SentenceEncoder(input_size=self.hidden_size, 
                                                hidden_size=self.hidden_size, 
                                                num_layers=self.hidden_layer, 
                                                bias=True, 
                                                dropout=self.dropout)
        
        self.predict_fc = nn.Sequential(nn.Dropout(p=self.dropout),
                                        nn.Linear(4 * 2 * self.hidden_size, self.hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(p=self.dropout),
                                        nn.Linear(self.hidden_size, self.n_classes))
        self.apply(init_model_weights)

    def forward(self, sentences1, sentences2):
        """
            sentences1 [batch, max_len]
            sentences2 [batch, max_len]
        """
        
        # get mask
        sentences1_mask = (sentences1 != self.padding_idx).long().to(self.device) # [batch_size, max_len]
        sentences2_mask = (sentences2 != self.padding_idx).long().to(self.device) # [batch_size, max_len]
        # input encoding
        sentences1_emb = self.emb(sentences1) # [batch_size, max_len, dim]
        sentences2_emb = self.emb(sentences2) # [batch_size, max_len, dim]
        sentences1_len = torch.sum(sentences1_mask, dim=-1).view(-1)# [batch_size]
        sentences2_len = torch.sum(sentences2_mask, dim=-1).view(-1)# [batch_size]
        #encoder
        s1_encoded = self.encoder_layer(sentences1_emb, sentences1_len) # [batch_size, max_len_q1, dim]
        s2_encoded = self.encoder_layer(sentences2_emb, sentences2_len) # [batch_size, max_len_q2, dim]
        # local inference
        # e_ij = a_i^Tb_j  (11)
        similarity_matrix = s1_encoded.bmm(s2_encoded.transpose(2, 1).contiguous()) # [batch_size, max_len_q1, max_len_q2]
        s1_s2_atten = masked_softmax(similarity_matrix, sentences2_mask)  # [batch_size, max_len_q1, max_len_q2]
        s2_s1_atten = masked_softmax(similarity_matrix.transpose(2, 1).contiguous(), sentences1_mask) # [batch_size, max_len_q2, max_len_q1]
        
        # eij * bj
        a_hat = weighted_sum(s1_encoded, s1_s2_atten, sentences1_mask) # [batch_size, max_len_q1, dim]
        b_hat = weighted_sum(s2_encoded, s2_s1_atten, sentences2_mask) # [batch_size, max_len_q2, dim]

        # Enhancement of local inference information
        # ma = [a¯; a~; a¯ − a~; a¯ a~];
        # mb = [b¯; b~; b¯ − b~; b¯ b~]
        m_a = torch.cat([s1_encoded, a_hat, s1_encoded - a_hat, s1_encoded * a_hat], dim=-1) # [batch_size, max_len_q1, 4 * dim]
        m_b = torch.cat([s2_encoded, b_hat, s2_encoded - b_hat, s2_encoded * b_hat], dim=-1)

        # 3.3 Inference Composition
        s1_projected = self.projection(m_a)  # [batch_size, max_len_q1, dim]
        s2_projected = self.projection(m_b)  # [batch_size, max_len_q2, dim]
        v_a = self.composition_layer(s1_projected, sentences1_len) # [batch_size, max_len_q1, dim]
        v_b = self.composition_layer(s2_projected, sentences2_len) # [batch_size, max_len_q2, dim]
        v_a_avg = torch.sum(v_a * sentences1_mask.unsqueeze(1).transpose(2, 1), dim=1)  \
                   / torch.sum(sentences1_mask, dim=1, keepdim = True) # q1_mask batch_size, 1, max_len_q1
        v_b_avg = torch.sum(v_b * sentences2_mask.unsqueeze(1).transpose(2, 1), dim=1) \
                   / torch.sum(sentences2_mask, dim=1, keepdim = True)
        v_a_max, _ = replace_masked(v_a, sentences1_mask, -1e7).max(dim=1) # [batch_size, dim]
        v_b_max, _ = replace_masked(v_b, sentences2_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1) # [batch_size, dim * 4]
        # predict
        logits = self.predict_fc(v)
        return logits