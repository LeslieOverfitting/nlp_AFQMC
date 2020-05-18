from datetime import timedelta
import numpy as np 
import torch 
import torch.nn as nn
import time

def get_time_diff(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))

def print_ans(acc, loss, report, confusion):
        msg = "Dev Loss:{0:>5.2}, Dev Acc:{1:>6.2%}"
        print(msg.format(loss, acc))
        print("Precision, Recall and F1-Score...")
        print(report)
        print("Confusion Matrix...")
        print(confusion)

def init_model_weights(module):
    """
    Initialise the weights of the inferSent model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
        batch [batch_size, max_seq_len, dim]
        sequences_lengths [batch_size, 1]
    """
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False) # 原先所在的位置
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

def get_mask(batch, sequences_lengths, max_len = 30):
    # return mask[batch, len] pad - 0 value - 1
    batch_size = batch.size()[0]
    mask = torch.ones(batch_size, max_len, dtype=torch.float)
    mask[batch[:, :max_len] == 0] == 0.0
    return mask