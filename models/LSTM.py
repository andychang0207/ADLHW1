from typing import Dict

import torch
from torch.nn import Embedding


class LSTM(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = torch.nn.LSTM(input_size=300,hidden_size=self.hidden_size,num_layers=self.num_layers,bidirectional=self.bidirectional,dropout=self.dropout)
        dirt = 2 if self.bidirectional else 1
        self.linear = torch.nn.Linear(in_features=dirt*self.hidden_size,out_features=self.num_class) 
        self.softmax = torch.nn.Softmax(dim=1) # softmax num class
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        # raise NotImplementedError

        # batch = [batch size, sent len]
        inputs = self.embed(batch)
        # inputs = [batch size, sent len, emb dim]
        inputs = inputs.permute(1,0,2)
        # inputs = [sent len, batch size, emb dim]
        
        output, _ = self.lstm(inputs,None)
        # output = [sent len, batch size, direction * hidden size]
        
        logits = self.linear(output[-1])
        # logits = [batch size, num class]
        
        # 回傳最後一個 time step 的 output [batch size, num class]
        return self.softmax(logits)
    
    def cal_loss(self, pred, target):
        """
        計算 loss
        """
        return self.criterion(pred, target)