import time
import sys
import os
from dataloader import *
from Transformer4Ranking.model import *
import paddle
from paddle import nn
from paddle.io import DataLoader
from metrics import evaluate_all_metric
from args import config
import numpy as np
# model = TransformerModel(
#     ntoken=config.ntokens,
#     hidden=config.emb_dim,
#     nhead=config.nhead,
#     nlayers=config.nlayers,
#     dropout=config.dropout,
#     mode='pretrain'
# )
config.train_datadir = "/home/hang-disk/桌面/WSDMCUP_BaiduPLM_Paddle/debug-data"
train_dataset = TrainDataset(config.train_datadir, max_seq_len=128, buffer_size=1000)
train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size)
model = TransformerModel(
    ntoken=config.ntokens, 
    hidden=config.emb_dim, 
    nhead=config.nhead, 
    nlayers=config.nlayers, 
    dropout=config.dropout,
    mode='pretrain'
)
print(model.token_encoder.weight.shape)
model.expand_emb()
# print(next(iter(train_data_loader)))
# for src_input, src_segment, src_padding_mask, click_label ,DisplayedTime_label,DwellingTime_label in train_data_loader:
#     # print("src_input:\n",src_input)
#     # print("src_segment:\n",src_segment)
#     # print("src_padding_mask:\n",src_padding_mask)
#     # print("click_label:\n",click_label)
#     # print("DisplayedTime_label:\n",DisplayedTime_label)
#     # print("DwellingTime_label:\n",DwellingTime_label)
#     # masked_src_input, mask_label = mask_data(src_input)
#     masked_src_input, mask_label = mask_data(src_input[ : ,:-config.num_of_nontext_feature+1])
#     print("--------------------start---------")
#     print('mask_label:\n',mask_label)
#     masked_src_nontext_input,mask_nontext_label=mask_data(src_input[ : , -config.num_of_nontext_feature:],mask_prob=0.5)
#     print('mask_nontext_label:\n',mask_nontext_label) 
#     mask_label = paddle.concat(x=[mask_label,mask_nontext_label],axis=1) # 横向堆叠
#     print('concat_mask_label:\n',mask_label)
#     print("-------------------end--------------") 
