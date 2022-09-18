import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleClassifier(nn.Module):
    def __init__(self, emb_size, vocab_size, pad_token_idxs,
            filters, filters_num, dropout=0.5, init=0.2, **kwargs):
        super(StyleClassifier, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.filters = filters # list of filter size
        self.filters_num = filters_num # list of filter num
        self.dropout_layer = nn.Dropout(p=dropout)
        self.IN_CHANNEL = 1
        assert len(self.filters) == len(self.filters_num)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        for i in range(len(self.filters)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.filters_num[i], self.emb_size * self.filters[i], stride=self.emb_size)
            setattr(self, f'conv_{i}', conv)
        self.affine = nn.Linear(sum(self.filters_num), 1)
        self.pad_token_idxs = pad_token_idxs

        if init:
            for p in self.parameters():
                p.data.uniform_(-init, init)
            for pad_token_idx in self.pad_token_idxs:
                self.embedding.weight.data[pad_token_idx].zero_()

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        max_sent_len = inp.size(1)
        embeddings = self.embedding(inp)
        x = embeddings.view(-1, 1, self.emb_size * max_sent_len)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), max_sent_len - self.filters[i] + 1)
                .contiguous().view(-1, self.filters_num[i])
            for i in range(len(self.filters))]
        x = torch.cat(conv_results, 1)
        x = self.dropout_layer(x)
        x = torch.sigmoid(self.affine(x).squeeze(1))
        return x

    def pad_embedding_grad_zero(self):
        for pad_token_idx in self.pad_token_idxs:
            self.embedding.weight.grad[pad_token_idx].zero_()

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
