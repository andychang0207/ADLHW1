import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
# import pandas as pd 不能用
import csv
import torch
from torch.utils.data import DataLoader

from dataset import TokenClsDataset
from models.BILSTMCRF.BILSTM import BiRnnCrf
from utils import Vocab

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2label: Dict[int, str] = {idx: tag for tag, idx in tag2idx.items()}
    data = json.loads(args.test_file.read_text())
    dataset = TokenClsDataset(data, vocab, tag2idx, args.max_len,'test')
    test_set = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False
        )
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = BiRnnCrf(
        9,
        embeddings,
        args.hidden_dim,
        args.dropout,
        args.num_layers
    ).to(args.device)

    ckpt = torch.load(args.ckpt_path)

    model.load_state_dict(ckpt)
    model.eval()
    preds = []
    index = []
    for batch, id in test_set:
        batch = batch.to(args.device)
        with torch.no_grad():
            _, pred = model(batch)
            # pred [batch size, seq size(each different)]
            # preds [datapoints size, seq size(each different)]
            preds.extend(pred)
            # index [datapoints size]
            index.extend(id)
    # preds [datapoints size, seq size]
    pred_labels = []
    for idx_list in preds:
        tags_list = []
        for idx in idx_list:
            tags_list.append(idx2label[idx])
        pred_labels.append(tags_list)
    # 不能用 pd
    # d = {'id':index,'tags':pred_labels}
    # df = pd.DataFrame(data=d)
    # df.to_csv(str(args.pred_file),index=False)
    with open(args.pred_file,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','tags'])
        for i, t in zip(index,pred_labels):
            s = " ".join(t)
            writer.writerow([i,s])
    print("Finish prediction")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/BILSTMCRF.pth"
    )
    parser.add_argument("--pred_file", type=Path, default="./pred_slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.)
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)