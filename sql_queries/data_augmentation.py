import random
from tkinter import N
random.seed(a=None)

table_dic = []
column_dic = []

class Augmentation:
    def __init__(self):
        self.sr_rate = 1.0
        self.big_aug = True
        self.rs_rate = 1.0

        self.sr_augment()
        self.rs_augment()
        self.final()

    def sr_augment(self):
        q = open('./modeling/train_data.txt', 'r')
        w = open('./modeling/temp/Augmented_data1.txt', 'w')

        q.readline()
        q.readline()

        while True:
            augmented = False

            cur_data = q.readline()
            if cur_data == '': break

            cur_data = cur_data.split('\t')
            query, label = cur_data
            
            label = label[:-1]

            query = query.split()

            for idx, val in enumerate(query):
                rand_val = random.random()
                if rand_val <= self.sr_rate:
                    if val == 'sum':
                        query[idx] = 'avg'
                        if self.big_aug == True:
                            w.write(' '.join(query) + '\t' + label + '\n')
                        augmented = True
                    elif val == 'avg':
                        query[idx] = 'sum'
                        if self.big_aug == True:
                            w.write(' '.join(query) + '\t' + label + '\n')
                        augmented = True

                    if val == 'count':
                        query[idx] = 'max'
                        if self.big_aug == True:
                            w.write(' '.join(query) + '\t' + label + '\n')
                        augmented = True
                    elif val == 'max':
                        query[idx] = 'count'
                        if self.big_aug == True:
                            w.write(' '.join(query) + '\t' + label + '\n')
                        augmented = True

            if augmented == True and self.big_aug == False:
                w.write(' '.join(query) + '\t' + label + '\n')

    def rs_augment(self):
        t = open('./modeling/meta_data/table_data.txt', 'r')
        c = open('./modeling/meta_data/column_data.txt', 'r')
        q = open('./modeling/train_data.txt', 'r')
        w = open('./modeling/temp/Augmented_data2.txt', 'w')

        while True:
            line = t.readline()
            if line == "": break
            
            _, table_alias = line.split()
            table_dic.append(table_alias)
        
        while True:
            line = c.readline()
            if line == "": break
            
            _, column_alias = line.split()
            column_dic.append(column_alias)

        # print("table dic :", table_dic)
        # print("column dic :", column_dic)

        q.readline()
        q.readline()

        while True:
            augmented = False

            cur_data = q.readline()
            if cur_data == '': break

            cur_data = cur_data.split('\t')
            query, label = cur_data
            
            label = label[:-1]

            query = query.split()

            col_set = False
            table_set = False

            col_list = []
            table_list = []

            for idx, val in enumerate(query):
                if val in column_dic:
                    if col_set == False:
                        col_set = True
                        col_list.append(val)
                    elif col_set == True:
                        col_list.append(val)
                elif val in table_dic:
                    if table_set == False:
                        table_set = True
                        table_list.append(val)
                    elif table_set == True:
                        table_list.append(val)                   
                elif col_set == True:
                    # print(" col before :", col_list)
                    random.shuffle(col_list)
                    # print("   ====> ", col_list)
                    for i in range(len(col_list)):
                        query[idx - len(col_list) + i] = col_list[i]
                    col_set = False
                    col_list.clear()
                elif table_set == True:
                    # print(" table before :", table_list)
                    random.shuffle(table_list)
                    # print("   ====> ", table_list)
                    for i in range(len(table_list)):
                        query[idx - len(table_list) + i] = table_list[i]
                    table_set = False
                    table_list.clear()

            rand_val = random.random()
            if rand_val <= self.rs_rate:
                w.write(' '.join(query) + '\t' + label + '\n')

    def final(self):
        q = open('./modeling/train_data.txt', 'r')
        aq1 = open('./modeling/temp/Augmented_data1.txt', 'r')
        aq2 = open('./modeling/temp/Augmented_data2.txt', 'r')
        w = open('./modeling/Augmented_train_data.txt', 'w')

        while True:
            cur = q.readline()
            if cur == '': break

            w.write(cur)

        w.write('\n')

        while True:
            cur = aq1.readline()
            if cur == '': break

            w.write(cur)

        while True:
            cur = aq2.readline()
            if cur == '': break

            w.write(cur)