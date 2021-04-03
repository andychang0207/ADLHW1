import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from dataset import TokenClsDataset
from models.BILSTMCRF.BILSTM import BiRnnCrf
from tqdm import trange
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utils import Vocab
from torch.utils.data import DataLoader
from seqeval.metrics import accuracy_score, classification_report, f1_score
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # 載入詞彙
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    # 載入 labels 和 index 其對應
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2label: Dict[int, str] = {idx: tag for tag, idx in tag2idx.items()}
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, TokenClsDataset] = {
        split: TokenClsDataset(split_data, vocab, tag2idx, args.max_len, split)
        for split, split_data in data.items()
    }
    train_set = DataLoader(
        dataset=datasets[TRAIN],
        batch_size=args.batch_size,
        collate_fn=datasets[TRAIN].collate_fn,
        shuffle=True)
    eval_set = DataLoader(
        dataset=datasets[DEV],
        batch_size=args.batch_size,
        collate_fn=datasets[DEV].collate_fn,
        shuffle=True)
    # 做好的詞彙和 word vector 對應
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(9527)
    torch.manual_seed(9527)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9527)
    
    model = BiRnnCrf(9,embeddings,args.hidden_dim, args.dropout, args.num_layers)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    model.to(args.device)
    loss_record = {'train':[],'dev':[]}
    acc_record = {'train':[],'dev':[]}
    best_val_loss = 10000
    the_last_loss = 10000
    patience = args.patience
    trigger_times = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # 訓練
        preds= []
        labels_list = []
        total_train_epoch_loss = 0
        model.train()
        for batch, labels, id, labels_wo_pad in train_set:
            model.zero_grad()
            batch, labels = batch.to(args.device), labels.to(args.device)
            loss = model.loss(batch, labels)
            loss.backward()
            optimizer.step()
            total_train_epoch_loss += loss.detach().cpu().item() * len(batch)
            with torch.no_grad():
                _, pred = model(batch)
                preds.extend(pred)
                labels_list.extend(labels_wo_pad)
        pred_labels = []
        for idx_list in preds:
            tags_list = []
            for idx in idx_list:
                tags_list.append(idx2label[idx])
            pred_labels.append(tags_list)
        join_train_acc = f1_score(labels_list,pred_labels)
        total_train_epoch_loss = total_train_epoch_loss / len(train_set.dataset)
        loss_record['train'].append(total_train_epoch_loss)
        acc_record['train'].append(join_train_acc)
        # validation
        preds= []
        labels_list = []
        model.eval()
        total_epoch_loss = 0
        with torch.no_grad():
            for batch, labels, id, labels_wo_pad in eval_set:
                batch, labels = batch.to(args.device), labels.to(args.device)
                loss = model.loss(batch,labels)
                _, pred = model(batch)
                preds.extend(pred)
                labels_list.extend(labels_wo_pad)
                total_epoch_loss += loss.detach().cpu().item() * len(batch)
            total_epoch_loss = total_epoch_loss / len(eval_set.dataset)
            loss_record['dev'].append(total_epoch_loss)
        pred_labels = []
        for idx_list in preds:
            tags_list = []
            for idx in idx_list:
                tags_list.append(idx2label[idx])
            pred_labels.append(tags_list)
        join_acc = f1_score(labels_list,pred_labels)
        subset_acc = accuracy_score(labels_list,pred_labels)
        acc_record['dev'].append(join_acc)
        print("epoch: {:2d}/{}, loss: {:5.2f}, val_loss: {:5.2f}, val_join_acc: {:5.2f}, val_subset_acc: {:5.2f}".format(epoch + 1, args.num_epoch, total_train_epoch_loss,total_epoch_loss, join_acc, subset_acc))
        print("-----------classification_report-------------")
        print(classification_report(labels_list, pred_labels, mode='strict', scheme=IOB2))
        print("---------------------------------------------")
        if total_epoch_loss < best_val_loss:
            best_val_loss = total_epoch_loss
            print("save model(epoch: {}, val_loss = {:.4f})".format(epoch + 1, best_val_loss))
            torch.save(model.state_dict(),str(args.ckpt_dir / (args.model_name+'.pth')))
        if total_epoch_loss > the_last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break
        else:
            trigger_times = 0
        the_last_loss = total_epoch_loss
    print("Finished training")
    plot_learning_curve(loss_record,'CRF Loss',str(args.fig_dir / (args.model_name+'_loss.jpg')),(args.model_name+' model'))
    plot_learning_curve(acc_record,'Join accuracy',str(args.fig_dir / (args.model_name+'_acc.jpg')),(args.model_name+' model'))



def plot_learning_curve(record,chart_type,file_path,title=''):
    ''' Plot learning curve of your model (train & dev loss) '''
    total_steps = len(record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(record['train']) // len(record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, record['train'], c='tab:red', label='train')
    plt.plot(x_2, record['dev'], c='tab:cyan', label='dev')
    plt.xlabel('epoch')
    plt.ylabel(chart_type)
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.savefig(file_path)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name",type=str,help="model name",default="BILSTMCRF")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset",
        default="./data/slot/"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file",
        default="./ckpt/slot/"
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        help="Directory to save the image.",
        default="./fig/slot/",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')

    parser.add_argument("--lr", type=float, default=1e-3)
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch",help="epoch", type=int, default=20)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    main(args)