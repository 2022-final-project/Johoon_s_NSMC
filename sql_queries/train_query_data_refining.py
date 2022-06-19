import glob

using_sql_word = ['select', 'from', 'where', 'order', 'group', 'by'
                , 'limit', 'when', 'then', 'case', 'having'
                , 'sum', 'avg', 'min', 'max', 'count', 'in', 'exists', 'like'
                , 'and', 'or', 'not'
                , 'date', 'month', 'year', 'asc', 'desc', 'on', 'end', 'if', 'else']
aggregate_words = ['sum', 'avg', 'min', 'max', 'count']
refining_target = [',', '(', ')', '\'']
alias_dic = {}
table_dic = {}
column_dic = {}

# sql file 들을 한줄화 / text화 한다.

class query_data_refining:
    def __init__(self):
        self.vocab_dic = {}

        self.sql_to_text()
        self.query_refining1()
        self.query_refining2()
        self.table_column_preprocessing()
        self.make_data_with_label()
        self.make_vocab()

    def sql_to_text(self):
        queries = glob.glob('./sql_queries/train_query_data/*.sql')
        sorted(queries)
        w = open('./sql_queries/temp/train_queries.txt', 'w')

        for query in queries:
            # ------------ query 한줄화 ------------
            q = open(query, 'r')

            str = q.readline()  # \o out.txt
            str = q.readline()  # SET random_page_cost = ~;

            query_on = False

            while True:
                str = q.readline()

                if 7 <= len(str) and str[:7] == "\\timing":
                    if query_on == False:
                        query_on = True
                        continue
                    elif query_on == True:
                        query_on = False
                        w.write('\n')
                        break

                if query_on == True:
                    w.write(str[:-1] + ' ') # query가 끝나는 줄이 아니면 '\n'가 없어지고,
                                            #   마지막 줄이면 ';'가 없어진다.
            #--------------------------------------

    def query_refining1(self):
        q = open('./sql_queries/temp/train_queries.txt', 'r')
        t = open('./modeling/meta_data/table_data.txt', 'r')
        c = open('./modeling/meta_data/column_data.txt', 'r')
        w = open('./sql_queries/temp/refined_train_queries1.txt', 'w')

        # table, column dictionary initialize
        while True:
            line = t.readline()
            if line == "": break
            
            table, table_alias = line.split()
            table_dic[table] = table_alias
        
        while True:
            line = c.readline()
            if line == "": break
            
            column, column_alias = line.split()
            column_dic[column] = column_alias

        query_list = []

        while True:
            query = q.readline()

            query = query.strip()
            query = query.replace("\t", "")

            query_list.append(query)

            if query == "": break
            word_list = query.split('[ \t]')

            bef_word = ""
            make_alias = False
            alias_target = ""

            for idx, word in enumerate(word_list):
                if word == "as":
                    make_alias = True
                    alias_target = bef_word
                    print("alais tar : ", alias_target)
                elif make_alias == True:
                    make_alias = False
                    if alias_target in column_dic:
                        alias_dic[word] = column_dic[alias_target]
                if idx < len(word_list) - 1:
                    w.write(word + ' ')
                else:
                    w.write(word[:-1] + '\n')
                    break
                bef_word = word

        print(alias_dic)

    def query_refining2(self):
        q = open('./sql_queries/temp/refined_train_queries1.txt', 'r')
        w = open('./sql_queries/temp/refined_train_queries2.txt', 'w')

        query_num = 1;

        while True:
            line = q.readline()
            if line == "": break
            
            line = line.strip()
            word_list = line.split()

            for idx, word in enumerate(word_list):
                while True:                                 # 겉을 strip
                    curWord = word
                    for refine_target in refining_target:
                        word = word.strip(refine_target)
                        if word == None:
                            word = ''
                            break
                    word = word.replace("(", " ")
                    if curWord == word:
                        break
                if 0 < len(word) and 48 <= ord(word[0]) and ord(word[0]) <= 57:   # 숫자로 시작하는 word인 경우
                    continue
                
                w.write(word + ' ')
            w.write('\n')
            query_num += 1;

    def table_column_preprocessing(self):
        q = open('./sql_queries/temp/refined_train_queries2.txt', 'r')
        w = open('./sql_queries/temp/refined_train_queries3.txt', 'w')

        while True:
            line = q.readline()

            if line == "": break

            line = line.strip()
            word_list = line.split()

            for idx, word in enumerate(word_list):
                word = word.lower()
                if word in table_dic:               # table info preprocessing
                    word = table_dic[word]
                elif word in column_dic:            # column info preprocessing
                    word = column_dic[word]
                elif word not in using_sql_word:    # not using sql word remove
                    word = ''

                if word == '': continue
                w.write(word + ' ')
            w.write('\n')

    def make_data_with_label(self):
        q = open('./sql_queries/temp/refined_train_queries3.txt', 'r')
        l = open('./modeling/train_labels.txt', 'r')
        w = open('./modeling/train_data.txt', 'w')

        w.write("query" + '\t' + "cost\n")

        while True:
            query = q.readline()
            label = l.readline()
            if query == '' or label == '': break
            query = query.strip()
            query = query.strip('\n')
            w.write(query + '\t' + label)

    def make_vocab(self):
        q = open('./sql_queries/temp/refined_train_queries3.txt', 'r')
        w = open('./modeling/vocab.txt', 'w')

        while True:
            query = q.readline()

            if query == '': break

            query = query.strip()
            word_list = query.split()

            for idx, word in enumerate(word_list):
                if word not in self.vocab_dic:
                    self.vocab_dic[word] = 1
                else:
                    self.vocab_dic[word] += 1

        self.vocab_dic = sorted(self.vocab_dic.items(), 
                                reverse = True, 
                                key = lambda item: item[1])

        w.write('[UNK]\n')
        w.write('[PAD]\n')
        w.write('[CLS]\n')
        w.write('[SEP]\n')

        for key in self.vocab_dic:
            print(key[0])
            w.write(key[0] + '\n')
        
        # vocab.txt padding
        initial_num = len(self.vocab_dic)

        vocab_size = 500
        for i in range(initial_num, vocab_size, 1):
            if i < vocab_size - 1:
                w.write('[unused' + str(i) + ']\n')
            else:
                w.write('[unused' + str(i) + ']')