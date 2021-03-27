from torch.nn import Embedding
from torch.utils.data import DataLoader
from utils import Vocab
import pickle
import torch
import json
from dataset import SeqClsDataset
with open("./cache/intent/vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)

# 載入 labels 和 index 其對應
with open('./cache/intent/intent2idx.json') as f:
    intent2idx = json.load(f)

with open('./data/intent/train.json') as f:
    data = json.load(f)
datasets =  SeqClsDataset(data,vocab,intent2idx,128)
train_dataset = DataLoader(datasets,128)
embbeddings = torch.load('./cache/intent/embeddings.pt')
embed = Embedding.from_pretrained(embbeddings,freeze=False)
max = 0
for datapoint in train_dataset:
    x = datapoint['intent']
    print(x)
    break
    # for s in x:
    #     s_list = s.split(' ')
    #     if max < len(s_list):
    #         max = len(s_list)
    # count = count + 1
    # if count == 1:
    #     continue
    # x_used = []
    # x = datapoint['text']
    # for s in x:
    #     s_list = s.split(' ')
    #     x_used.append(s_list)
    # x_new = vocab.encode_batch(x_used)
    # x_new = torch.LongTensor(x_new)
    # inputs = embed(x_new)
    # print(inputs.size())
    # break
# print(max)