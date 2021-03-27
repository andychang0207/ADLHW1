from typing import Dict

import torch
from torch.nn import Embedding


class RNN(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = torch.nn.RNN(input_size=300,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=dropout)
        dirt = 2 if bidirectional else 1
        self.linear = torch.nn.Linear(hidden_size=dirt*hidden_size,num_class=num_class) 
        

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError

        # batch = [batch size, sent len]
        inputs = self.embed(batch)
        # inputs = [batch size, sent len, emb dim]
        inputs = inputs.permute(1,0,2)
        # inputs = [sent len, batch size, emb dim]
        output, hidden = self.rnn(inputs)
        # output = [sent len, batch size, direction * hidden size]

        logits = self.linear(output)
        # logits = [sent len, batch size, num class]

        return logits[-1]

