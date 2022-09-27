import torch
import torch.nn as nn
import torch.nn.functional as F


class TextStyleClassifier(nn.Module):
    """ Classic TextCNN classification model.
    """
    def __init__(self, vocab_size=None, pad_idx=0, embed_size=100,
            filters=[2, 3, 5], filters_num=[10, 20, 30], dropout=0.5, init_weight=0.2, **kwargs):
        super(TextStyleClassifier, self).__init__()
        self.vocab_size, self.embed_size = vocab_size, embed_size
        self.filters = filters # list of filter size
        self.filters_num = filters_num # list of filter num
        assert len(self.filters) == len(self.filters_num)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=pad_idx)
        self.max_kernel_size, self.pad_idx = max(self.filters), pad_idx
        for i in range(len(self.filters)):
            conv = nn.Conv1d(1, self.filters_num[i], self.embed_size * self.filters[i], stride=self.embed_size)
            setattr(self, f'conv_{i}', conv)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.affine = nn.Linear(sum(self.filters_num), 1)

        if init_weight is not None and init_weight > 0:
            for p in self.parameters():
                p.data.uniform_(-init_weight, init_weight)
            self.embedding.weight.data[pad_idx].zero_()


    def get_conv(self, i):
        return getattr(self, f'conv_{i}')


    def forward(self, inputs):
        max_sent_len = inputs.size(1)
        if max_sent_len < self.max_kernel_size:
            paddings = inputs.new_full((inputs.size(0), self.max_kernel_size - max_sent_len), self.pad_idx)
            inputs = torch.cat([inputs, paddings], dim=1)
            max_sent_len = self.max_kernel_size
        embeddings = self.embedding(inputs)
        x = embeddings.view(-1, 1, self.embed_size * max_sent_len)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), max_sent_len - self.filters[i] + 1)
                .contiguous().view(-1, self.filters_num[i])
            for i in range(len(self.filters))]
        x = torch.cat(conv_results, 1)
        x = self.dropout_layer(x)
        x = torch.sigmoid(self.affine(x).squeeze(1))
        return x