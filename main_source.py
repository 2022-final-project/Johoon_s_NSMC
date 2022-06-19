from sql_queries.train_query_data_refining import query_data_refining as TRAIN
from sql_queries.test_query_data_refining import query_data_refining as TEST
from sql_queries.data_augmentation import Augmentation

TRAIN()
# sql query들을 train_query_data 폴더에서 하나하나 받아온 뒤 train_data로 쓰이기 적합한 형태로 바꾼다.
# train_data.txt 를 만든다.

TEST()
# sql query들을 test_query_data 폴더에서 하나하나 받아온 뒤 test_data로 쓰이기 적합한 형태로 바꾼다.
# test_data.txt 를 만든다.

Augmentation()
# train_data들을 Augmentation을 진행한다.
# Augmented_train_data.txt 를 만든다.