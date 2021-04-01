import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from torch.utils.data import DataLoader
import csv
from customerror import InvalidModelName
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from models.RNN import RNN
from models.LSTM import LSTM
from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # 載入詞彙
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # 載入 labels 和 index 其對應
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # 
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len, split)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
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
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)
    if args.model_name == "RNN":
        model = RNN(embeddings,args.hidden_size,args.num_layers,args.dropout,args.bidirectional,150).to(args.device)
    elif args.model_name == "LSTM":
        model = LSTM(embeddings,args.hidden_size,args.num_layers,args.dropout,args.bidirectional,150).to(args.device)
    else:
        raise InvalidModelName("No such model "+ args.model_name)
    print(model)
    # TODO: init optimizer
    optimizer = getattr(torch.optim,args.opt)(model.parameters(),lr=args.lr)

    min_ce = 1000.
    loss_record = {'train':[],'dev':[]}
    acc_record = {'train':[],'dev':[]}
    # 測試用
    loss_dev_record = []
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # change model to train mode
        total_epoch_loss = 0
        total_epoch_acc = 0
        model.train()
        for batch, labels, id in train_set:
            # batch_tmp = []
            # datapoints = batch['text']
            # labels = batch['intent']
            # for s in datapoints:
            #     s_list = s.split(' ')
            #     batch_tmp.append(s_list)
            # batch_new = vocab.encode_batch(batch_tmp,to_len=28)
            # batch_new = torch.LongTensor(batch_new)

            # set gradient to zero
            optimizer.zero_grad()
            # move data to device (cpu/cuda)
            batch, labels = batch.to(args.device), labels.to(args.device)
            # forward pass (compute output)
            pred = model(batch)
            # compute loss
            ce_loss = model.cal_loss(pred, labels)
            # compute acc
            acc = cal_accuracy(pred,labels)
            # compute gradient (backpropagation)
            ce_loss.backward()
            # update model with optimizer
            optimizer.step()
            # record loss
            total_epoch_loss += ce_loss.detach().cpu().item() * len(batch)
            # record acc
            total_epoch_acc += acc.detach().cpu().item() * len(batch)

        total_epoch_loss = total_epoch_loss / len(train_set.dataset)
        total_epoch_acc = total_epoch_acc / len(train_set.dataset)
        loss_record['train'].append(total_epoch_loss)
        acc_record['train'].append(total_epoch_acc)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # change model to evalutation mode
        model.eval()
        total_epoch_loss = 0
        total_epoch_acc = 0
        for batch, labels, id in eval_set:
            batch, labels = batch.to(args.device), labels.to(args.device)
            # 當我們在做evaluating的時候（不需要計算導數），我們可以將推斷（inference）的代碼包裹在with torch.no_grad():之中，以達到暫時不追踪網絡參數中的導數的目的，總之是為了減少可能存在的計算和內存消耗
            with torch.no_grad():
                pred = model(batch)
                ce_loss = model.cal_loss(pred,labels)
                acc = cal_accuracy(pred,labels)
            total_epoch_loss += ce_loss.detach().cpu().item() * len(batch)
            total_epoch_acc += acc.detach().cpu().item() * len(batch)
        total_epoch_loss = total_epoch_loss / len(eval_set.dataset)
        total_epoch_acc = total_epoch_acc / len(eval_set.dataset)
        loss_record['dev'].append(total_epoch_loss)
        acc_record['dev'].append(total_epoch_acc)
        if total_epoch_loss < min_ce:
            # Save model if your model improved
            min_ce = total_epoch_loss
            print('Saving model (epoch = {:4d}, val_loss = {:.4f})'
                .format(epoch + 1, min_ce))
            torch.save(model.state_dict(), str(args.ckpt_dir / (args.model_name+'2.pth')))  # Save model to specified path
    print('Finished training')
    plot_learning_curve(loss_record,'Cross Entropy Loss',str(args.fig_dir / (args.model_name+'2_loss.jpg')),(args.model_name+' model'))
    plot_learning_curve(acc_record,'Accuracy',str(args.fig_dir / (args.model_name+'2_acc.jpg')),(args.model_name+' model'))
    # 測試 eval 預測出甚麼
    # with open('./test2_train.csv','w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['epoch','loss'])
    #     for i,p in enumerate(loss_record['train']):
    #         writer.writerow([i,p])
    # with open('./test2_dev.csv','w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['epoch','loss'])
    #     for i,p in enumerate(loss_dev_record):
    #         writer.writerow([i,p])
    # out = []
    # id_out = []
    # model.eval()
    # for batch, labels, id, intent in eval_set:
    #     batch = batch.to(args.device)
    #     with torch.no_grad():
    #         pred = model(batch)
    #         out.append(pred.detach().cpu())
    #         for i in id:
    #             id_out.append(i)
    # out = torch.cat(out, dim = 0).numpy()
    # print(len(out))
    # with open('./test.csv','w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['id', 'tested_positive'])
    #     for i , p in zip(id_out,out):
    #         writer.writerow([i,p])
    # TODO: Inference on test set

# 畫圖用
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

def cal_accuracy(pred, target):
    classes = torch.argmax(pred,1)
    return torch.mean((classes==target).float())



# input 用

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name",type=str,help="model name",default="LSTM")
    parser.add_argument(
        "--rand_seed",
        type=int,
        help="set a random seed for reproducibility.",
        default=9527,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        help="Directory to save the image.",
        default="./fig/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--opt", type=str, default='Adam')
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch",help="epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    main(args)
