from typing import Iterable, List


# 出現過的詞彙之class
class Vocab:
    # pad 每個batch的單字數長度要一致，pad是用來補齊的
    # unk 若輸入字典沒出現過的，則用unk代替
    PAD = "[PAD]"
    UNK = "[UNK]"

    # token2idx : {
    #   "[PAD]" : 0,
    #   "[UNK]" : 1,
    #   "第一個字" : 2,
    #   ....
    # }
    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    # 叫出 PAD 的 ID
    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    # 叫出 UNK 的 ID
    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    # 叫出所有字的 list
    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    # 叫出某個字的 ID 若沒有則回傳 UNK 的 ID
    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    # 把字的 list encode 成 ID
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    # 把每個 batch encode 成 ID
    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids

# 用 pad 補齊 batch 的單字數長度
def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds
