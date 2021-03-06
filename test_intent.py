import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
# import pandas as pd 不能用
import torch
from torch.utils.data import DataLoader
import csv
from dataset import SeqClsDataset
from models.LSTM import LSTM
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2label: Dict[int, str] = {idx: intent for intent, idx in intent2idx.items()}
    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len,'test')
    # TODO: crecate DataLoader for test dataset
    test_set = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False
        )
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = LSTM(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    ).to(args.device)
    

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    # TODO: predict dataset
    model.eval()
    preds = []
    index = []
    for batch, id in test_set:
        batch = batch.to(args.device)
        with torch.no_grad():
            pred = model(batch)
            # [batch size, num classes]
            pred = torch.argmax(pred,dim = 1)
            # pred = [batch size]
            preds.append(pred.detach().cpu())
            index.extend(id)
    # preds = [num epoch, batch size]
    preds = torch.cat(preds, dim = 0).numpy()
    # preds = [datapoints size]
    # index = [batch size]
    # transform preds idx to labels
    pred_labels = []
    for idx in preds:
        pred_labels.append(idx2label[idx])
    # TODO: write prediction to file (args.pred_file)
    # d = {'id':index,'intent':pred_labels}
    with open(args.pred_file,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'intent'])
        for i, p in zip(index,pred_labels):
            writer.writerow([i,p])
    # 不能用 pd
    # df = pd.DataFrame(data=d)
    # df.to_csv(str(args.pred_file),index=False)
    print("Finish prediction")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/LSTM2.pth"
    )
    parser.add_argument("--pred_file", type=Path, default="./pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

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
