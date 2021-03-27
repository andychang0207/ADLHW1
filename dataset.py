from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

# 將 dataset 轉成 pytorch 型別 : Dataset
class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
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

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        raise NotImplementedError

    # 輸入 label 得到對應 index
    def label2idx(self, label: str):
        return self.label_mapping[label]

    # 輸入 index 得到對應 label
    def idx2label(self, idx: int):
        return self._idx2label[idx]
