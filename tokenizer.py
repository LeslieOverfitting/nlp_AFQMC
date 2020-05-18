import collections

class Tokenizer:
    def __init__(self, vocab_list):
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        self.vocab = self.load_vocab(vocab_list) # key-word value index
    
    def load_vocab(self, vocab_list):
        '''
            return: 
                    vocab: key 字词 value index
        '''
        vocab = collections.OrderedDict()
        vocab[self.pad] = 0
        vocab[self.unk] = 1
        index = 2
        for token in vocab_list:
            token = token.strip()
            vocab[token] = index
            index += 1
        return vocab
    
    def token_to_id(self, token):
        idx = self.vocab.get(token)
        if idx is not None:
            return idx
        else:
            return self.vocab[self.unk]

    def tokens_to_id(self, tokens):
        ids_list = list(map(self.token_to_id, tokens))
        return ids_list