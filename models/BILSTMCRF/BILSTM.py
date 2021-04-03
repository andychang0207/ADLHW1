import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .CRF import CRF


class BiRnnCrf(nn.Module):
    def __init__(self, tagset_size, embeddings, hidden_dim, dropout , num_rnn_layers=1, rnn="lstm"):
        super(BiRnnCrf, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(300, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, dropout=dropout, batch_first=True)
        self.crf = CRF(hidden_dim, self.tagset_size)

    def __build_features(self, sentences):
        
        # 標記哪些是真的有值的，哪些是 pad 的
        # sentences [batch size, 35(seq size)]
        # masks [batch size, 35(seq size)]
        masks = sentences.gt(0)
        # 做 embedding
        # embeds = [batch size, 35(seq size), 300(embed dim)]
        embeds = self.embedding(sentences.long())
        
        # 各 sentence 的長度
        # seq_length [batch size]
        seq_length = masks.sum(1)
        # sorted_seq_length 各 sentence 長度由大排到小，[batch size]
        # perm_idx 排列後 index，[batch size]
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        
        # 照 sentence 長度由大排到小的 embedding [batch size(seq num), 35(seq size), 300(embed size)]
        embeds = embeds[perm_idx, :]
        
        # 處理經過 padding 的 data，會去除 padding 且打包成 torch nn 能處理的形式
        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        # 訓練
        packed_output, _ = self.rnn(pack_sequence)
        # 解包，還原成 padding 的樣子
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        # 順序排回來
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        # 算 CRF 的 loss
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq