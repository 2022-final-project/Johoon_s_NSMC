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

# train_data.txt 와 test_data.txt 를 읽어온다.
train_txt = open('./modeling/Augmented_train_data.txt', 'r')
train = pd.read_csv(train_txt, sep='\t')

# test_txt = open('./test_data_temp.txt', 'r')
# test = pd.read_csv(test_txt, sep='\t')

queries = train['query']    # train_data.txt 의 query 들 양 옆으로 "[CLS]", "[SEP]" 를 붙인다.
queries = ["[CLS] " + str(query[:-1]) + " [SEP]" for query in queries]

NUM_LABELS = 6

labels_before_preprocessing = train['cost']     
labels = []

# label preprocessing
''' label값이 011010인 경우 [0, 1, 1, 0, 1, 0]의 6차원 vector가 된다.
'''

for cost in labels_before_preprocessing:
    cost_str = str(cost)
    if len(cost_str) == 5: cost_str = "0" + cost_str
    elif len(cost_str) == 4 : cost_str = "00" + cost_str
    elif len(cost_str) == 3 : cost_str = "000" + cost_str
    elif len(cost_str) == 2 : cost_str = "0000" + cost_str
    elif len(cost_str) == 1 : cost_str = "00000" + cost_str
    elif len(cost_str) == 0 : cost_str = "000000"

    label_temp = [0, 0, 0, 0, 0, 0]
    if cost_str[0] == '1': label_temp[0] = 1
    if cost_str[1] == '1': label_temp[1] = 1
    if cost_str[2] == '1': label_temp[2] = 1
    if cost_str[3] == '1': label_temp[3] = 1
    if cost_str[4] == '1': label_temp[4] = 1
    if cost_str[5] == '1': label_temp[5] = 1
    labels.append(label_temp)

# ------------------------------------------- Data preProcessing -------------------------------------------
''' 1. Tokenizing
'''

# ["[CLS] select c1 from t1 [SEP]"]

tokenizer = BertTokenizer.from_pretrained("./modeling/vocab.txt")       # 구현된 vocab.txt file로 tokenizer를 구현한다.
tokenized_queries = [tokenizer.tokenize(query) for query in queries]    # 구현된 tokenizer로 query들을 모두 tokenizing 한다.

# ['[CLS]', 'select', 'c1', 'from', 't1', '[SEP]']

MAX_LEN = 60

input_ids = [tokenizer.convert_tokens_to_ids(tokenized_query) for tokenized_query in tokenized_queries]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

# [2, 6, 13, 5, 28, 3, 0, 0, 0, ...] --> MAX_LEN 길이를 채우게끔 padding 진행

attention_masks = []

for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# 0이 아닌 곳을 1로 masking 한다. query가 있는 부분만 1값을 가지게 된다.

# -------------------------------- train_data.txt 로 train/validation set 을 얻는다. --------------------------

RAND_SEED = random.randint(1, 3000)
VALIDATION_RATE = 0.2
BATCH_SIZE = 16

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels, 
                                                                                    random_state=RAND_SEED, 
                                                                                    test_size=VALIDATION_RATE)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=RAND_SEED, 
                                                       test_size=VALIDATION_RATE)

train_inputs = torch.tensor(train_inputs).float()
train_labels = torch.tensor(train_labels).float()
train_masks = torch.tensor(train_masks).float()
validation_inputs = torch.tensor(validation_inputs).float()
validation_labels = torch.tensor(validation_labels).float()
validation_masks = torch.tensor(validation_masks).float()          

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

device = torch.device("cpu")

# ---------------------------------- model 생성 -------------------------------------

# 에폭수
EPOCHS = 200

config = BertConfig.from_pretrained('bert-base-uncased', problem_type="regression", hidden_dropout_prob=0)
config.num_labels = NUM_LABELS

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = NUM_LABELS)
# print(model.parameters) -> 확인 결과: (classifier): Linear(in_features=768, out_features=6, bias=True)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # 학습률
                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * EPOCHS

# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# accuracy를 구하는 function 1 이다.
''' target label    == [1, 1, 0, 1, 0, 0]
    predicted label == [0, 0, 0, 1, 0, 0]
    이라면 이 set에 대한 accuracy는 100%이다.
'''
def flat_accuracy(preds, labels):
    cnt = 0
    total_cnt = 0

    for idx, pred in enumerate(preds):
        # print(" label :", labels[idx][:])
        print(" pred sigmoid 전 :", pred[:])
        total_cnt += 6

        maxIdx = 0
        for i in range(0,6,1):
            if pred[maxIdx] < pred[i]:
                # print(" i :", i, " / ", pred[maxIdx], ", ", pred[i])
                maxIdx = i

        # pred에서 가장 큰 값을 가지는 label의 index
        # print(" pred max idx :", maxIdx)
        if labels[idx][maxIdx] == 1:    # label에서 maxIdx에 해당하는 값이 1이면 성공!
            cnt += 6
            # print("i == ", idx, " / cnt : ", cnt, " / total cnt :", total_cnt)

        print(" pred sigmoid 후 :", pred[:])
        print(" label           :", labels[idx][:])

    return cnt / total_cnt


# accuracy를 구하는 function 2 이다.
''' target label    == [1, 1, 0, 1, 0, 0]
    predicted label == [0, 1, 1, 1, 0, 1]
    이라면 이 set에 대한 accuracy는 50%이다.
'''
def flat_exact_accuracy(preds, labels):
    cnt_exact = 0
    cnt = 0
    total_cnt = 0

    for idx, pred in enumerate(preds):
        print(" Exact pred sigmoid 전 :", pred[:])

        total_cnt += 6

        maxVal = 0

        for i in range(0, 6, 1):
            if 0 < i and pred[maxVal] < pred[i]: maxVal = i # Accuracy

            if pred[i] < 0: pred[i] = 0.0
            else: pred[i] = 1.0
            if pred[i] == labels[idx][i] :
                cnt_exact += 1

        if labels[idx][maxVal] == 1: cnt += 6
    
        print(" Exact pred sigmoid 후 :", pred[:])
        print(" pred max index        :", maxVal)
        print(" label                 :", labels[idx][:])

    return cnt_exact / total_cnt, cnt / total_cnt

def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

# ---------------------- 긴 modeling 부분 --------------------

# 재현을 위해 랜덤시드 고정
seed_val = RAND_SEED
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()

# epoch 당 loss 그래프를 그리기 위한 리스트
train_loss=[]
val_loss=[]
train_exact_acc=[]
train_acc=[]
val_acc=[]

def plt_loss_graph(loss, dataset_name):
    plt.plot(loss)
    plt.axis([0, EPOCHS, 0, max(loss)+1])
    plt.xlabel("Epochs")
    plt.ylabel(dataset_name)
    plt.ylim([0.0, max(loss) + 0.1])
    plt.show()

def plt_graph(val, dataset_name):
    plt.plot(val)
    plt.axis([0, EPOCHS, 0, max(val)+1])
    plt.xlabel("Epochs")
    plt.ylabel(dataset_name)
    plt.ylim([min(val) - 0.05, 1.0])
    plt.show()

# 에폭만큼 반복
for epoch_i in range(0, EPOCHS):
    
    # ========================================
    #               Training
    # ========================================
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
    print('Training...')

    t0 = time.time()    # 시작 시간 설정
    total_loss = 0      # 로스 초기화
    model.train()       # 훈련모드로 변경
        
    # data_loader 에서 batch 만큼 반복하여 data를 가져온다.
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if True:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        # b_labels = b_labels.squeeze(0)

        # Forward 수행                
        outputs = model(b_input_ids,
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        # print("outputs : ", outputs)  # Output: loss, logits, hidden_states, attentions

        loss = outputs[0]           # 로스 구함
        total_loss += loss.item()   # 총 로스 계산
        loss.backward()             # Backward 수행으로 그래디언트 계산

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 그래디언트 클리핑
        optimizer.step()                                        # 그래디언트를 통해 가중치 파라미터 업데이트
        scheduler.step()                                        # 스케줄러로 학습률 감소
        model.zero_grad()                                       # 그래디언트 초기화

    avg_train_loss = total_loss / len(train_dataloader)         # 평균 로스 계산

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    train_loss.append(float(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    #시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy, eval_exact_accuracy = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        # batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # 출력 로짓 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
        
        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_exact_accuracy, tmp_eval_accuracy = flat_exact_accuracy(logits, b_labels)
        eval_exact_accuracy += tmp_eval_exact_accuracy
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Exact Accuracy: {0:.2f}".format(eval_exact_accuracy/nb_eval_steps))
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    train_exact_acc.append(eval_exact_accuracy/nb_eval_steps)
    train_acc.append(eval_accuracy/nb_eval_steps)
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")
plt_loss_graph(train_loss, "train loss")
plt_graph(train_exact_acc, "validation exact accuracy")
plt_graph(train_acc, "validation accuracy")
# plt_loss_graph(val_loss, "validation loss")

model.save_pretrained('./modeling/model.pt')