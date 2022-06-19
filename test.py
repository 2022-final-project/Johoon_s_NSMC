import tensorflow as tf
import torch

from tensorflow import keras
from transformers import pipeline
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
import datetime
import random

tokenizer = BertTokenizer.from_pretrained("./modeling/vocab.txt")       # 구현된 vocab.txt file로 tokenizer를 구현한다.
model = BertForSequenceClassification.from_pretrained('./modeling/model.pt')

device = torch.device("cpu")

# ---------------------- 새로운 문장 테스트 --------------------

# 입력 데이터 변환
def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids) # 데이터 타입: torch.int64
    masks = torch.tensor(attention_masks) # 데이터 타입: torch.float32

    return inputs, masks

    # 문장 테스트
def test_sentences(sentences):

    # 평가모드로 변경
    model.eval()
    # print("======== 최종 Result ========\n=== input data: ", sentences)

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)
    # print("=== convert_input_data 결과 \ninputs: ", inputs, "\nmasks: ", masks)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
            
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
        # print("=== outputs: ", outputs)

    # 출력 로짓 구함
    logits = outputs[0]
    
    ret = [0, 0, 0, 0, 0, 0]

    # print("logits :", logits)

    for val in logits:
        if val[0] > 0: ret[0] = 1
        if val[1] > 0: ret[1] = 1
        if val[2] > 0: ret[2] = 1
        if val[3] > 0: ret[3] = 1
        if val[4] > 0: ret[4] = 1
        if val[5] > 0: ret[5] = 1

    print(" ==> result : ", ret)

# ——————————— 새로운 문장 테스트 입력 ——————————
# print(" 추가적인 문장 test\n")

test_q = open('./modeling/test_data.txt', 'r')
test_queries = []

while True:
    q = test_q.readline()

    if q == '': break
    test_queries.append(q)

# for q in test_queries:
#     print(" query :", q[:-1])
#     test = test_sentences(q[:-1])
logits_test1 = test_sentences(['select c9 sum c14 c15 c33 c36 from t1 t4 t2 where c7 and c1 c30 and c9 c29 and c33 date and c19 date group by c9 c33 c36 order by desc c33 asc'])   
logits_test2 = test_sentences(['select avg c14 from t2 t5 t6 where c19 date and c38 c54 and c38 c10 and c41 like and c44 and c13 select avg c13 from t2 where c10 c38'])   
logits_test3 = test_sentences(['select c52 c48 c26 c38 c40 c49 c51 c53 from t5 t8 t6 t3 where c38 c54 and c47 c55 and c43 and c42 like and c50 c25 and c43 and and c57 select min c57 from t6 t3 t7 t8 where c38 c54 and c47 c55 and c50 c25 and c27 c59 and c60 order by c52 c26 c48 c38'])   
logits_test4 = test_sentences(['select c9 sum c14 c15 count c33 from t1 t4 t2 where c7 and c1 c30 and c9 c29 and c33 date group by c9 c33 order by desc c33 asc'])