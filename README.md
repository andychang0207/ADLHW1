# Sample Code for Homework 1 ADL NTU 109 Spring

## Installation
```shell
pip install -r requirements.txt
```

## Download
```shell
bash download.sh
```

## train model
### Intent detection
```shell
python train_intent.py --model_name LSTM4 --dropout 0.5 --num_epoch 450
```

### Slot tagging
```shell
python train_slot.py --model_name BILSTMCRF --num_epoch 20
```
