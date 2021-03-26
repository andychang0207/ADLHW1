import json
import logging
import pickle
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from typing import List, Dict

import torch
from tqdm.auto import tqdm

from utils import Vocab

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# words 字和出現次數對應，vocab_size 不同字的總數量，output_dir 輸出路徑，glove_path glove的路徑位置
def build_vocab(
    words: Counter, vocab_size: int, output_dir: Path, glove_path: Path
) -> None:
    # 出現過不同的字之集合
    common_words = {w for w, _ in words.most_common(vocab_size)}
    # 出現過的詞彙之class
    vocab = Vocab(common_words)
    vocab_path = output_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocab saved at {str(vocab_path.resolve())}")

    glove: Dict[str, List[float]] = {}
    logging.info(f"Loading glove: {str(glove_path.resolve())}")
    with open(glove_path) as fp:
        row1 = fp.readline()
        # if the first row is not header (數字開始和數字結尾)
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        # 用空白分開，第一欄是字，其他是vector
        for i, line in tqdm(enumerate(fp)):
            cols = line.rstrip().split(" ")
            word = cols[0]
            vector = [float(v) for v in cols[1:]]

            # 不再我們 dataset 中的詞彙就略過
            # skip word not in words if words are provided
            if word not in common_words:
                continue
            # glove 這個 dict 存放字和其 glove word vector 的對應
            glove[word] = vector
            glove_dim = len(vector)

    # 確保每個字向量維度相同，以及詞彙大小小於規定值
    assert all(len(v) == glove_dim for v in glove.values())
    assert len(glove) <= vocab_size

    num_matched = sum([token in glove for token in vocab.tokens])
    logging.info(
        f"Token covered: {num_matched} / {len(vocab.tokens)} = {num_matched / len(vocab.tokens)}"
    )
    # 將 dataset 出現的字變成 glove 的 word vector，若不在 glove 中則用隨機機率 vector 代替
    embeddings: List[List[float]] = [
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
        for token in vocab.tokens
    ]
    # 變成 pytorch 的 tensor 形式
    embeddings = torch.tensor(embeddings)
    embedding_path = output_dir / "embeddings.pt"
    torch.save(embeddings, str(embedding_path))
    logging.info(f"Embedding shape: {embeddings.shape}")
    logging.info(f"Embedding saved at {str(embedding_path.resolve())}")


def main(args):
    # 隨機產生不存在 glove 中的 vector 用 (line 68)
    seed(args.rand_seed)

    intents = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        # train 和 evals 的 labels 集合
        intents.update({instance["intent"] for instance in dataset})

        # train 和 evals 出現過的不同字和其出現次數
        words.update(
            [token for instance in dataset for token in instance["text"].split()]
        )

    # labels 和 index 的對應
    intent2idx = {tag: i for i, tag in enumerate(intents)}
    intent_tag_path = args.output_dir / "intent2idx.json"
    intent_tag_path.write_text(json.dumps(intent2idx, indent=2))
    logging.info(f"Intent 2 index saved at {str(intent_tag_path.resolve())}")

    build_vocab(words, args.vocab_size, args.output_dir, args.glove_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # 資料的路徑
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    # glove 的路徑
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    # random seed 預設 13
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    # preprocess完輸出的路徑
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/intent/",
    )
    # 出現過詞彙數量 預設 10000
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=10_000,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
