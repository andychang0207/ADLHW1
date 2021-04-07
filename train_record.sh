python train_slot.py --model_name BILSTMCRF --num_epoch 20
python train_slot.py --model_name BILSTMCRF_test --num_epoch 10 --lr 0.1 --hidden_dim 300
python train_slot.py --model_name BILSTMCRF_test --num_epoch 150
python train_slot.py --model_name BILSTMCRF_test --num_epoch 150 --lr 0.001 --patience 20
python train_slot.py --model_name BILSTMCRF_L2 --weight_decay 1e-6 --num_epoch 20 --patience 10
python train_slot.py --batch_size 10 --model_name test_test
python train_slot.py --batch_size 10 --model_name BILSTMCRF_B_10_L2 --weight_decay 1e-6
python train_slot.py --batch_size 10 --model_name BILSTMCRF_B_10_L2_d_0_8_e_40 --weight_decay 1e-6 --dropout 0.8 --num_epoch 40
python train_intent.py --model_name LSTM4 --dropout 0.5 --num_epoch 300
python train_intent.py --model_name LSTM4 --dropout 0.5 --num_epoch 450