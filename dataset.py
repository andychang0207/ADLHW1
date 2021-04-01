from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab

# 將 dataset 轉成 pytorch 型別 : Dataset
class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode: str
    ):
        # data : 資料 eg. 
        # [{
        #   'id' : ID,
        #   'intent' : label, 
        #   'text' : 句子
        # },
        # ....]
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

    # datapoint 大小
    def __len__(self) -> int:
        return len(self.data)
    # 輸入 index 能得到 datapoint
    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    # class 數量
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        # print(samples)
        # print(type(samples))
        # print(len(samples))
        # raise NotImplementedError
        batch_tmp = []
        datapoints = [ d['text'] for d in samples]
        id = [ d['id'] for d in samples]
        # 處理 input
        for s in datapoints:
            s_list = s.split(' ')
            batch_tmp.append(s_list)
        batch_new = self.vocab.encode_batch(batch_tmp,to_len=28)
        batch_new = torch.LongTensor(batch_new)
        if self.mode == "train" or self.mode == "eval":
            labels = [ d['intent'] for d in samples]
            labels_tmp = []
            for intent in labels:
                labels_tmp.append(self.label2idx(intent))
            labels_new = torch.LongTensor(labels_tmp)
            return batch_new, labels_new, id
        else:
            return batch_new, id

    # 輸入 label 得到對應 index
    def label2idx(self, label: str):
        return self.label_mapping[label]

    # 輸入 index 得到對應 label
    def idx2label(self, idx: int):
        return self._idx2label[idx]


# 將 dataset 轉成 pytorch 型別 : Dataset
class TokenClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode: str
    ):
        # data : 資料 eg. 
        # [{
        #   'id' : ID,
        #   'intent' : label, 
        #   'text' : 句子
        # },
        # ....]
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

    # datapoint 大小
    def __len__(self) -> int:
        return len(self.data)
    # 輸入 index 能得到 datapoint
    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    # class 數量
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        batch_tmp = [ d['tokens'] for d in samples]
        id = [ d['id'] for d in samples]
        # 處理 input
        batch_new = self.vocab.encode_batch(batch_tmp,to_len=35)
        batch_new = torch.LongTensor(batch_new)
        if self.mode == "train" or self.mode == "eval":
            # labels:[batch size, seq size]
            labels = [ d['tags'] for d in samples]
            labels_tmp = []
            for tags in labels:
                new_tags = []
                for tag in tags:
                    new_tags.append(self.label2idx(tag))
                new_tags = new_tags + [8] * (35 - len(new_tags))
                labels_tmp.append(new_tags)
            labels_new = torch.LongTensor(labels_tmp)
            return batch_new, labels_new, id
        else:
            return batch_new, id

    # 輸入 label 得到對應 index
    def label2idx(self, label: str):
        return self.label_mapping[label]

    # 輸入 index 得到對應 label
    def idx2label(self, idx: int):
        return self._idx2label[idx]
