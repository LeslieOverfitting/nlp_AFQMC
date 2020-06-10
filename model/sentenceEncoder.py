import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import sort_by_seq_lens, get_mask

class SentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bias=True, dropout=0.0, bidirectional=True):
        super(SentenceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self._encoder = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            bias=self.bias, 
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, sequences_batch, sequnces_lengths):
        max_len = sequences_batch.shape[1]
        sorted_batch, sorted_length, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequnces_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_length, batch_first=True)
        outputs, _ = self._encoder(packed_batch) # hidden = None
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=max_len) # [batch_size, max_len, dim]
        # restore order
        reorder_outputs = outputs.index_select(0, restoration_idx)
        return reorder_outputs
