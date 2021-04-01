from torch.nn import Embedding
from torch.utils.data import DataLoader
from utils import Vocab, pad_to_len
import pickle
import torch
import json
from dataset import SeqClsDataset
from models.RNN import RNN
# with open("./cache/intent/vocab.pkl", "rb") as f:
#     vocab: Vocab = pickle.load(f)

# # 載入 labels 和 index 其對應
# with open('./cache/intent/intent2idx.json') as f:
#     intent2idx = json.load(f)

with open('./data/slot/train.json') as f:
    data = json.load(f)
max = 0
for item in data:
    if max < len(item['tokens']):
        max = len(item['tokens'])
print(max)
# datasets =  SeqClsDataset(data,vocab,intent2idx,128)
# train_dataset = DataLoader(datasets,128)
# embbeddings = torch.load('./cache/intent/embeddings.pt')
# embed = Embedding.from_pretrained(embbeddings,freeze=False)
# max = 0
# for datapoint in train_dataset:
#     x = datapoint['intent']
#     print(x)
#     break
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
# with open("./cache/intent/vocab.pkl", "rb") as f:
#         vocab: Vocab = pickle.load(f)


# with open("./cache/intent/intent2idx.json", "rb") as f:
#     intent2idx: Dict[str, int] = json.load(f)
# inputs = []
# labels = []
# x = "how long should i cook steak for"
# y = "cook_time"
# y = intent2idx[y]
# labels.append(y)
# labels = torch.LongTensor(labels)
# x = x.split(' ')
# x = vocab.encode(x)
# x = pad_to_len(x,28,0)
# inputs.append(x)
# inputs = torch.LongTensor(inputs)
# inputs, labels = inputs.to('cuda'), inputs.to('cuda')
# embbeddings = torch.load('./cache/intent/embeddings.pt')

# model = RNN(
#     embeddings,
#     512,
#     2,
#     0.1,
#     True,
#     150,
# ).to('cuda')
# model.eval()
# with torch.no_grad():
#     pred = model(inputs)
#     out = 
# ckpt = torch.load(str(args.ckpt_path / 'model.pth'))
# # load weights into model
# model.load_state_dict(ckpt)

# x = torch.randn(2,3)
# y = torch.randn(2,3)
# z = torch.randn(2,3)
# p = [x,y,z]
# print(p)
# print(torch.cat(p,dim=0))
