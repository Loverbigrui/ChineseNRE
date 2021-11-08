#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zh
# data:  4:09 下午
# ide: PyCharm
# coding:utf8
import numpy as np
import pickle
import sys
import codecs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable
from bilstm_attention import BiLSTM_ATT
import os

os.environ["OMP_NUM_THREADS"] = "1"
# with open('./data/engdata_train.pkl', 'rb') as inp:
with open('./data/train.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)
    train = pickle.load(inp)
    labels = pickle.load(inp)
    position1 = pickle.load(inp)
    position2 = pickle.load(inp)

# with open('./data/engdata_test.pkl', 'rb') as inp:
with open('./data/test.pkl', 'rb') as inp:
    test = pickle.load(inp)
    labels_t = pickle.load(inp)
    position1_t = pickle.load(inp)
    position2_t = pickle.load(inp)
# print(len(test), len(train))


EPOCHS = 100
config = dict()
config['EMBEDDING_SIZE'] = len(word2id) + 1
config['EMBEDDING_DIM'] = 100
config['POS_SIZE'] = 82
config['POS_DIM'] = 25
config['HIDDEN_DIM'] = 200
config['TAG_SIZE'] = len(relation2id)
config['BATCH'] = 128
config["pretrained"] = True

learning_rate = 0.0005

embedding_pre = []
# embedding_pre = []
# if len(sys.argv) == 2 and sys.argv[1] == "pretrained":
    # print "use pretrained embedding"
    # config["pretrained"] = True
word2vec = {}
with codecs.open('vec.txt', 'r', 'utf-8') as input_data:
    for line in input_data.readlines():
        word2vec[line.split()[0]] = map(eval, line.split()[1:])

unknow_pre = []
unknow_pre.extend([1] * 100)
embedding_pre.append(unknow_pre)  # wordvec id 0
for word in word2id:
    if word in word2vec:
        embedding_pre.append(word2vec[word])
    else:
        embedding_pre.append(unknow_pre)

embedding_pre = np.asarray(embedding_pre)
# if len(sys.argv) == 2 and sys.argv[1] == "pretrained":

    # config["pretrained"] = True
# word2vec = {}
# with codecs.open('vec.txt', 'r', 'utf-8') as input_data:
#     for line in input_data.readlines():
#         # print(len(line.split()[1:]))
#         word2vec[line.split()[0]] = line.split()[1:]
#
# unknow_pre = []
# unknow_pre.extend([1.0] * config['EMBEDDING_DIM'])
# embedding_pre.append(unknow_pre)  # wordvec id 0
# for word in id2word:
#     if str(word) in word2vec.keys():
#         embedding_pre.append(word2vec[word])
#     else:
#         embedding_pre.append(unknow_pre)
# embedding_pre = np.asarray(embedding_pre)
# embedding_pre = np.asarray(embedding_pre, dtype=float)
# embedding_pre = embedding_pre.astype(float)
model = BiLSTM_ATT(config, embedding_pre)
# model = torch.load('model/model_epoch20.pkl')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(reduction='mean')

train = torch.LongTensor(train[:len(train) - len(train) % config['BATCH']])
position1 = torch.LongTensor(position1[:len(train) - len(train) % config['BATCH']])
position2 = torch.LongTensor(position2[:len(train) - len(train) % config['BATCH']])
labels = torch.LongTensor(labels[:len(train) - len(train) % config['BATCH']])
train_datasets = D.TensorDataset(train, position1, position2, labels)
train_dataloader = D.DataLoader(train_datasets, config['BATCH'], True, num_workers=8)

test = torch.LongTensor(test[:len(test) - len(test) % config['BATCH']])
position1_t = torch.LongTensor(position1_t[:len(test) - len(test) % config['BATCH']])
position2_t = torch.LongTensor(position2_t[:len(test) - len(test) % config['BATCH']])
labels_t = torch.LongTensor(labels_t[:len(test) - len(test) % config['BATCH']])
test_datasets = D.TensorDataset(test, position1_t, position2_t, labels_t)
test_dataloader = D.DataLoader(test_datasets, config['BATCH'], True, num_workers=8)

for epoch in range(EPOCHS):
    for sentence, pos1, pos2, tag in train_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence, pos1, pos2)
        tags = Variable(tag)
        loss = criterion(y, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_t = 0
    total_t = 0
    count_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_right = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for sentence, pos1, pos2, tag in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence, pos1, pos2)
        y = np.argmax(y.data.numpy(), axis=1)
        for y1, y2 in zip(y, tag):
            count_predict[y1] += 1
            count_total[y2] += 1
            if y1 == y2:
                count_right[y1] += 1

    precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(count_predict)):
        if count_predict[i] != 0:
            precision[i] = float(count_right[i]) / count_predict[i]

        if count_total[i] != 0:
            recall[i] = float(count_right[i]) / count_total[i]

    precision = sum(precision) / len(relation2id)
    recall = sum(recall) / len(relation2id)
    print('{}轮'.format(epoch + 1))
    print(precision)
    print(recall)

    if epoch % 20 == 0:
        model_name = "./model/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, model_name)

torch.save(model, "./model/model_01.pkl")
